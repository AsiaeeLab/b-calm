"""Shared output and numerical comparison helpers for experiment drivers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from utils import log


def write_table(
    path: Path, caption: str, label: str, columns: str, header: str, rows: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    lines += [row + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def compare_frames(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    *,
    digits: int | None = None,
) -> dict:
    """Compare keyed rows exactly, optionally at the source CSV precision."""
    a = actual.copy().fillna("").sort_values(keys).reset_index(drop=True)
    b = expected.copy().fillna("").sort_values(keys).reset_index(drop=True)
    if len(a) != len(b):
        raise AssertionError(f"row count: {len(a)} != {len(b)}")
    differences = {}
    for col in b:
        if col not in a:
            raise AssertionError(f"missing column: {col}")
        if pd.api.types.is_numeric_dtype(b[col]) and pd.api.types.is_numeric_dtype(
            a[col]
        ):
            x, y = a[col].to_numpy(), b[col].to_numpy()
            if digits is not None:
                x = np.array([float(f"{v:.{digits}g}") for v in x])
            equal = np.array_equal(x, y)
            diff = float(np.max(np.abs(x - y))) if len(x) else 0.0
        else:
            equal = a[col].astype(str).equals(b[col].astype(str))
            diff = 0.0 if equal else "text mismatch"
        if not equal:
            differences[col] = diff
    result = {"exact": not differences, "rows": len(a), "differences": differences}
    log(f"reproduction: {result}")
    return result
