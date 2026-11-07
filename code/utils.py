from __future__ import annotations

import csv
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
FAILURES_DIR = RESULTS_DIR / "failures"
RUN_TIMES = RESULTS_DIR / "run_times.json"


def ensure_dirs() -> None:
    for path in (CODE_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, FAILURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


class Timer:
    def __init__(self, label: str):
        self.label = label
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def finish(self) -> float:
        seconds = self.elapsed()
        log(f"{self.label} finished in {format_seconds(seconds)}")
        return seconds


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.2f}h"


def seed_list(n_seeds: int, offset: int = 0) -> list[int]:
    return [offset + i for i in range(int(n_seeds))]


def sigma_label(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if math.isinf(float(value)):
        return "inf"
    return f"{float(value):.6g}"


def sigma_value(value: float | str) -> float:
    if isinstance(value, str) and value.lower() == "inf":
        return math.inf
    return float(value)


def read_done(path: Path, key_cols: Sequence[str]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    done: set[tuple[str, ...]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(col not in reader.fieldnames for col in key_cols):
            return set()
        for row in reader:
            done.add(tuple(str(row[col]) for col in key_cols))
    return done


def append_csv_rows(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                val = row.get(key, "")
                if isinstance(val, float):
                    if math.isinf(val):
                        clean[key] = "inf"
                    elif math.isnan(val):
                        clean[key] = "nan"
                    else:
                        clean[key] = f"{val:.10g}"
                else:
                    clean[key] = val
            writer.writerow(clean)
    return len(rows)


def dump_failure(seed: int, config: Mapping[str, object], exc: BaseException) -> Path:
    ensure_dirs()
    label = f"{timestamp().replace(' ', '_').replace(':', '-')}_seed{seed}"
    path = FAILURES_DIR / f"{label}.json"
    payload = {
        "timestamp": timestamp(),
        "seed": seed,
        "config": dict(config),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _portable(value: object) -> object:
    """Store paths inside the repository relative to its root."""
    if isinstance(value, (str, Path)) and Path(value).is_absolute():
        try:
            return str(Path(value).resolve().relative_to(ROOT))
        except ValueError:
            return Path(value).name
    return value


def record_wallclock(experiment: str, seconds: float, details: Mapping[str, object] | None = None, *, out_dir: Path | None = None) -> None:
    ensure_dirs()
    output = RUN_TIMES if out_dir is None else Path(out_dir) / "run_times.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        payload = json.loads(output.read_text())
    else:
        payload = {}
    payload[experiment] = {
        "timestamp": timestamp(),
        "seconds": float(seconds),
        "formatted": format_seconds(seconds),
        "details": {key: _portable(value) for key, value in dict(details or {}).items()},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))


def progress(index: int, total: int, start_time: float, label: str, every: int = 25) -> None:
    if index == 1 or index == total or index % every == 0:
        elapsed = time.time() - start_time
        rate = elapsed / max(index, 1)
        eta = rate * (total - index)
        log(f"{label}: {index}/{total}, elapsed {format_seconds(elapsed)}, eta {format_seconds(eta)}")
