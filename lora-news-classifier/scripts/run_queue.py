"""Run a queue of training runs one after another, unattended (e.g. overnight).

    python scripts/run_queue.py stage5

Each run is `python -m src.train ...` followed by scripts/verify_reload.py on
its adapter. A failed run is logged and the queue moves on. Everything goes to
outputs/queue_<name>.log.

On Windows the script asks the OS not to sleep while it is running (CPU load
alone does not count as "activity"); the normal power settings come back by
themselves when it exits. Closing the laptop lid can still suspend it.
"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

QUEUES = {
    # Stage 5: the two base models, same recipe and LoRA settings.
    # bloom first: distilgpt2 is ~3.7 h on CPU (13 s/step, ~190 tokens per headline).
    "stage5": [
        ["--base", "bloom-560m", "--targets", "attn", "--rank", "8", "--lr", "1e-3"],
        ["--base", "distilgpt2", "--targets", "attn", "--rank", "8", "--lr", "1e-3"],
    ],
}

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake(on: bool) -> None:
    if sys.platform == "win32":
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


def run_name(args: list[str]) -> str:
    opts = dict(zip(args[::2], args[1::2]))
    return f"{opts['--base']}_{opts['--targets']}_r{opts['--rank']}"


def main():
    queue = sys.argv[1]
    log_path = Path(f"outputs/queue_{queue}.log")
    with log_path.open("a", encoding="utf-8") as log:
        def note(msg):
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
            log.flush()

        keep_awake(True)
        try:
            for args in QUEUES[queue]:
                name = run_name(args)
                note(f"start {name}: {' '.join(args)}")
                train = subprocess.run([sys.executable, "-u", "-m", "src.train", *args, "--name", name],
                                       stdout=log, stderr=subprocess.STDOUT)
                if train.returncode != 0:
                    note(f"FAILED {name} (exit {train.returncode})")
                    continue
                check = subprocess.run([sys.executable, "-u", "scripts/verify_reload.py", f"outputs/runs/{name}"],
                                       stdout=log, stderr=subprocess.STDOUT)
                note(f"done {name} (reload check exit {check.returncode})")
            note("queue finished")
        finally:
            keep_awake(False)


if __name__ == "__main__":
    main()
