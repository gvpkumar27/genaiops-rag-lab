import json
import time
import uuid
from pathlib import Path

RUNS_DIR = Path("data/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def log_event(event: dict) -> None:
    event.setdefault("event_id", uuid.uuid4().hex)
    event["ts"] = time.time()
    with (RUNS_DIR / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
