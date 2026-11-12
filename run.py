import argparse
from pathlib import Path
import subprocess
import sys

from genevolve_repro.common import ROOT, WORLD_SIZE, required_env


def distributed(subcommand, work, *arguments):
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}", str(ROOT / "run.py"),
        "--work", str(work), subcommand, *arguments,
    ], cwd=ROOT, check=True)


def evaluation_arguments(args):
    result = ["--split", args.split]
    if args.policy is not None:
        result.extend(["--policy", str(args.policy.resolve())])
    if args.max_cases is not None:
        result.extend(["--max-cases", str(args.max_cases)])
    return result


def eval_flags(parser):
    parser.add_argument("--split", choices=["bench", "rl-val", "wise"], default="bench")
    parser.add_argument("--policy", type=Path, help="HF export (default: WORK/evolve/final)")
    parser.add_argument("--max-cases", type=int, help="Limit a live evaluation for a pilot")


def execute(args, work):
    if args.command in {"all", "evolve", "eval"}:
        keys = ("WISE_API_KEY",) if getattr(args, "split", None) == "wise" else ("JUDGE_BASE_URL", "JUDGE_API_KEY")
        for key in ("SERPER_API_KEY", *keys):
            required_env(key)
    if args.command in {"prepare", "all"}:
        from genevolve_repro.data import prepare
        prepare(work)
    if args.command in {"sft", "all"}:
        from genevolve_repro.sft import run
        run(work)
    if args.command in {"evolve", "all"}:
        steps = getattr(args, "steps", None)
        distributed("_evolve", work, *([] if steps is None else ["--steps", str(steps)]))
    if args.command in {"eval", "all"}:
        flags = evaluation_arguments(args) if args.command == "eval" else []
        distributed("_eval", work, *flags)


def main():
    parser = argparse.ArgumentParser(description="Reproduce GenEvolve with Qwen3-VL-2B on four GPUs")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="download pinned models and datasets")
    commands.add_parser("sft", help="train or resume multimodal SFT")
    for name in ("evolve", "_evolve"):
        stage = commands.add_parser(name)
        stage.add_argument("--steps", type=int, help="RL updates (default: one pass over the train split)")
    for name in ("eval", "_eval"):
        eval_flags(commands.add_parser(name))
    commands.add_parser("all", help="prepare, run SFT and self-evolution, then evaluate")
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == "_evolve":
        from genevolve_repro.evolve import worker
        worker(work, args.steps)
    elif args.command == "_eval":
        from genevolve_repro.evaluate import worker
        policy = args.policy.resolve() if args.policy else work / "evolve" / "final"
        worker(work, policy, args.split, args.max_cases)
    else:
        from filelock import FileLock, Timeout
        work.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(work / ".run.lock", timeout=0):
                execute(args, work)
        except Timeout as error:
            raise SystemExit(f"Another run is using {work}.") from error


if __name__ == "__main__":
    main()
