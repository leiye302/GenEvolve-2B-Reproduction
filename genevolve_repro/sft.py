from pathlib import Path
import subprocess
import sys

from .common import ROOT, SFT, WORLD_SIZE, model_ready, read_json, write_json


SFT_FINAL_STEP = 550


def launch_config(work, resume=None):
    config = read_json(ROOT / "configs" / "sft.json")
    config.update(
        model_name_or_path=str(work / "base"),
        dataset_dir=str(work / "data" / SFT),
        media_dir=str(work / "data" / SFT),
        output_dir=str(work / "sft"),
        deepspeed=str(ROOT / ".deps" / "llamafactory" / "examples" / "deepspeed" / "ds_z3_config.json"),
    )
    if resume is not None:
        config["resume_from_checkpoint"] = str(resume)
    return config


def _can_resume(path):
    if not (path / "trainer_state.json").is_file() or not (path / "latest").is_file():
        return False
    native = path / (path / "latest").read_text(encoding="utf-8").strip()
    if native.parent != path or not native.is_dir():
        return False
    return len(list(native.glob("*optim_states.pt"))) >= WORLD_SIZE and bool(
        list(native.glob("*model_states.pt"))
    )


def run(work):
    out = work / "sft"
    state = out / "trainer_state.json"
    if model_ready(out) and state.is_file() and read_json(state)["global_step"] == SFT_FINAL_STEP:
        print(f"SFT complete at {out}", flush=True)
        return
    if not (work / "prepared.json").is_file():
        raise RuntimeError("Run python run.py prepare first.")
    checkpoints = sorted(
        (p for p in out.glob("checkpoint-*") if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()),
        key=lambda p: int(p.name.removeprefix("checkpoint-")),
    )
    resume = next((p for p in reversed(checkpoints) if _can_resume(p)), None)
    if checkpoints and resume is None:
        raise RuntimeError(f"No complete ZeRO-3 checkpoint under {out}. Use another --work directory.")
    # LLaMA-Factory reads YAML, of which a JSON object is a compatible subset.
    args = work / "sft-launch.yaml"
    config = launch_config(work, resume)
    if not Path(config["deepspeed"]).is_file():
        raise RuntimeError("Run bash scripts/setup.sh before starting SFT.")
    from .sft_probe import verify_labels

    labels = verify_labels(work / "base", work / "data" / SFT, config)
    write_json(work / "sft-label-probe.json", labels)
    write_json(args, config)
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
         f"--nproc-per-node={WORLD_SIZE}", "--module", "llamafactory.launcher", str(args)],
        cwd=ROOT, check=True,
    )
    if not model_ready(out) or not state.is_file() or read_json(state)["global_step"] != SFT_FINAL_STEP:
        raise RuntimeError(f"SFT exited without a complete step-{SFT_FINAL_STEP} HF export at {out}.")
