from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .actor_text import decode_actor_response


ACTION = re.compile(r"<tool_call>.*?</tool_call>|<answer>.*?</answer>", re.DOTALL)
RAW_REFERENCE = re.compile(r"https?://|IMG_\d+", re.IGNORECASE)


def action_indices(tokenizer, tokens):
    """Locate generated tool-call and final-program tokens, excluding reasoning."""
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    spans = [match.span() for match in ACTION.finditer(text)]
    if not spans:
        return []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    eligible = {
        i for i, (start, end) in enumerate(offsets)
        if end > start and any(start < right and end > left for left, right in spans)
    }
    if encoded["input_ids"] == tokens:
        return sorted(eligible)
    # A generated BPE sequence need not be the canonical encoding of its text.
    matched = SequenceMatcher(a=encoded["input_ids"], b=tokens, autojunk=False)
    return [
        block.b + offset for block in matched.get_matching_blocks()
        for offset in range(block.size) if block.a + offset in eligible
    ]


@dataclass
class Turn:
    messages: list
    prefix: list[int]
    tokens: list[int]
    action: list[int]
    old: torch.Tensor | None = None
    reference: torch.Tensor | None = None
    teacher: torch.Tensor | None = None
    distill: torch.Tensor | None = None


@dataclass
class Trajectory:
    result: object
    turns: list[Turn]


def valid_program(result):
    from pathlib import Path

    return (
        result.termination == "answer"
        and bool(result.gen_prompt.strip())
        and not RAW_REFERENCE.search(result.gen_prompt)
        and 1 <= len(result.reference_images) <= 2
        and all(ref.get("local_path") and Path(ref["local_path"]).is_file()
                for ref in result.reference_images)
    )


class ActorScorer(nn.Module):
    """Teacher-force exactly the continuation sampled in a rollout turn."""

    def __init__(self, model, projection_chunk=128):
        super().__init__()
        self.model = model
        self.projection_chunk = projection_chunk

    def forward(self, input_ids, prefix_length, temperature):
        hidden = self.model.model(
            input_ids=input_ids, use_cache=False, return_dict=True
        ).last_hidden_state[0, prefix_length - 1:-1]
        targets = input_ids[0, prefix_length:]

        def token_logp(states, token_ids):
            logits = self.model.lm_head(states).float() / temperature
            return -F.cross_entropy(logits, token_ids, reduction="none")

        parts = []
        for states, ids in zip(hidden.split(self.projection_chunk), targets.split(self.projection_chunk)):
            value = (
                checkpoint(token_logp, states, ids, use_reentrant=False)
                if torch.is_grad_enabled()
                else token_logp(states, ids)
            )
            parts.append(value)
        return torch.cat(parts)


def score_turn(scorer, prefix, tokens, temperature, device):
    if not prefix or not tokens:
        raise ValueError("A replay turn must have both context and sampled tokens.")
    ids = torch.tensor([prefix + tokens], dtype=torch.long, device=device)
    return scorer(ids, len(prefix), temperature)


def tokenize_history(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def teacher_prefix(tokenizer, messages, guidance):
    teacher_messages = [dict(message) for message in messages]
    system = teacher_messages[0]["content"]
    tool_section = "# Tools"
    before, found, after = system.partition(tool_section)
    if not found:
        raise ValueError("The pinned agent's system prompt has no # Tools section.")
    teacher_messages[0]["content"] = (
        f"{before}\n<visual_experience>\n{guidance}\n</visual_experience>\n\n"
        f"{found}{after}"
    )
    return tokenize_history(tokenizer, teacher_messages)


class BudgetExhausted(Exception):
    pass


def make_agent(model, tokenizer, device, config, **tools):
    from .common import load_upstream

    load_upstream()
    from genevolve.agent import GenEvolveAgent
    from genevolve.tools import ImageSearchTool, WebTextSearchTool

    search_errors = []

    class TrackBackendFailures:
        def _post(self, payload):
            try:
                return super()._post(payload)
            except Exception as error:
                search_errors.append(error)
                raise

    class TrainingTextSearch(TrackBackendFailures, WebTextSearchTool):
        pass

    class TrainingImageSearch(TrackBackendFailures, ImageSearchTool):
        pass

    if "text_search_tool" not in tools:
        tools["text_search_tool"] = TrainingTextSearch()
    if "image_search_tool" not in tools:
        tools["image_search_tool"] = TrainingImageSearch()

    class TrainingAgent(GenEvolveAgent):
        def collect(self, prompt):
            self.turns = []
            self.generated = 0
            self.failure = None
            search_errors.clear()
            result = self.run(prompt)
            if self.failure is not None:
                raise RuntimeError("Actor generation failed.") from self.failure
            if search_errors:
                raise RuntimeError("The Serper service failed during rollout.") from search_errors[0]
            return Trajectory(result, self.turns)

        @torch.no_grad()
        def _call_llm(self, messages):
            prefix = tokenize_history(tokenizer, messages)
            available = min(
                config["max_turn_tokens"],
                config["max_response_tokens"] - self.generated,
                config["context_tokens"] - len(prefix),
            )
            if not self.turns and len(prefix) > config["max_initial_tokens"]:
                raise BudgetExhausted("The initial token budget was exceeded.")
            if available < 1:
                raise BudgetExhausted("The generation token budget was exhausted.")
            try:
                ids = torch.tensor([prefix], dtype=torch.long, device=device)
                output = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=available,
                    do_sample=True,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=0,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                tokens = output[0, len(prefix):].tolist()
                if not tokens:
                    raise RuntimeError("The actor returned zero tokens.")
                self.turns.append(Turn(
                    [dict(message) for message in messages], prefix, tokens,
                    action_indices(tokenizer, tokens),
                ))
                self.generated += len(tokens)
                return decode_actor_response(tokenizer, tokens)
            except Exception as error:
                self.failure = error
                raise

    return TrainingAgent(
        max_rounds=config["max_rounds"],
        max_tokens_per_round=config["max_turn_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_prompt_length=config["max_initial_tokens"],
        max_response_length=config["max_response_tokens"],
        n_max_reference_images=2,
        **tools,
    )
