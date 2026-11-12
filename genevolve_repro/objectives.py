import math
from statistics import fmean, stdev

import torch


def advantages(rewards, epsilon=1e-6):
    if len(rewards) < 2:
        raise ValueError("GRPO requires multiple samples for the same prompt.")
    average = fmean(rewards)
    scale = stdev(rewards) + epsilon
    return [(reward - average) / scale for reward in rewards]


def k3(log_ratio):
    # Outside this range the cap on k3 has already saturated.
    log_ratio = log_ratio.clamp(-20, 20)
    return (torch.expm1(-log_ratio) + log_ratio).clamp(0, 10)


def kl_reward(reward, turns, coef):
    penalty = sum(k3(turn.old - turn.reference).sum().item() for turn in turns)
    return reward - coef * penalty


def choose_distillation_tokens(turns, fraction):
    """Select the largest old-student/teacher disagreement across a trajectory."""
    candidates = []
    for number, turn in enumerate(turns):
        turn.distill = torch.zeros(len(turn.tokens), dtype=torch.bool)
        if turn.teacher is not None:
            candidates.extend(
                (abs(float(turn.old[i] - turn.teacher[i])), number, i) for i in turn.action
            )
    count = math.ceil(fraction * len(candidates))
    for _, number, index in sorted(candidates, reverse=True)[:count]:
        turns[number].distill[index] = True
    return count


def loss_terms(new, turn, advantage, cfg):
    """Return per-turn sums. The caller averages over complete trajectories."""
    old = turn.old.to(new.device)
    log_ratio = (new - old).clamp(-20, 20)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1 - cfg["clip_low"], 1 + cfg["clip_high"])
    grpo = -torch.minimum(advantage * ratio, advantage * clipped).sum()
    sdl = new.new_zeros(())
    if turn.teacher is not None and turn.distill is not None:
        mask = turn.distill.to(new.device)
        target = turn.teacher.to(new.device)[mask].detach()
        weighted = ratio[mask].clamp(max=cfg["sdl_ratio_cap"]) * k3(new[mask] - target)
        sdl = weighted.clamp(max=cfg["sdl_token_cap"]).sum()
    return grpo, sdl
