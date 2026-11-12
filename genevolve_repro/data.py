from dataclasses import dataclass
from pathlib import Path

from .common import BENCH, RL, SFT, jsonl, model_ready, relative_image, source_lock, write_json


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    gt: Path | None


def examples(folder, split):
    rows = []
    ids = set()
    for row in jsonl(folder / f"{split}.jsonl"):
        sample_id = str(row["id"])
        prompt = row.get("prompt") or row.get("question")
        if not isinstance(prompt, str) or not prompt.strip() or not sample_id or sample_id in ids:
            raise ValueError(f"Invalid or repeated example {sample_id!r} in {folder}/{split}")
        gt = relative_image(folder, row["gt_image"])
        rows.append(Example(sample_id, prompt, gt))
        ids.add(sample_id)
    return rows


def sft_registry():
    return {
        f"genevolve_{split}": {
            "file_name": f"{split}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        for split in ("train", "eval")
    }


def verify_data(work):
    folder = work / "data"
    sft = folder / SFT
    counts = {}
    images = set()
    for split, expected in (("train", 8800), ("eval", 200)):
        count = 0
        for row in jsonl(sft / f"{split}.jsonl"):
            messages = row["messages"]
            roles = [message["role"] for message in messages]
            if len(roles) < 3 or roles != ["system"] + ["user", "assistant"] * ((len(roles) - 1) // 2):
                raise ValueError(f"Bad SFT role sequence in {split} row {count}")
            if sum(m["content"].count("<image>") for m in messages) != len(row["images"]):
                raise ValueError(f"Bad SFT image placeholders in {split} row {count}")
            images.update(row["images"])
            count += 1
        if count != expected:
            raise ValueError(f"SFT {split} has {count} rows, expected {expected}")
        counts[f"sft_{split}"] = count
    for image in sorted(images):
        relative_image(sft, image)
    for dataset, split, count in ((RL, "train", 2575), (RL, "eval", 600), (BENCH, "test", 594)):
        actual = len(examples(folder / dataset, split))
        if actual != count:
            raise ValueError(f"{dataset}/{split} has {actual} rows, expected {count}")
        counts[f"{dataset}/{split}"] = actual
    counts["sft_images"] = len(images)
    write_json(sft / "dataset_info.json", sft_registry())
    return counts


def prepare(work):
    from huggingface_hub import snapshot_download

    lock = source_lock()
    for name in ("base", "embedding"):
        model = lock[name]
        print(f"Preparing {model['repo']} at {model['revision'][:7]}", flush=True)
        snapshot_download(
            model["repo"], revision=model["revision"], local_dir=work / name,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt", "*.model"],
        )
    dataset = lock["data"]
    print(f"Preparing {dataset['repo']} at {dataset['revision'][:7]}", flush=True)
    snapshot_download(
        dataset["repo"], revision=dataset["revision"], repo_type="dataset", local_dir=work / "data",
        allow_patterns=[f"{folder}/{pattern}" for folder in (SFT, RL, BENCH)
                        for pattern in ("*.jsonl", "images/*")],
    )
    if not model_ready(work / "base") or not model_ready(work / "embedding"):
        raise RuntimeError("The pinned model snapshot is incomplete.")
    counts = verify_data(work)
    write_json(work / "prepared.json", {"sources": lock, "counts": counts})
    print(counts, flush=True)
