import os
from pathlib import Path

import pandas as pd
from filelock import FileLock


# Append rows to a shared CSV, deduplicating by key. Safe with concurrent processes.
# `key` may be a column name or a list of columns (e.g. ["chunk_id", "round"]).
# Key columns missing from older files (written before a column existed) are
# filled with `missing_key_default`, so a new "round" column reads as round 0.
def append_dedup_csv(output_path, new_rows: pd.DataFrame, key: str | list[str] = "chunk_id",
                     missing_key_default=0) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [key] if isinstance(key, str) else list(key)

    with FileLock(str(output_path) + ".lock"):
        if output_path.exists():
            existing = pd.read_csv(output_path, encoding="utf-8")
            for k in keys:
                if k not in existing.columns:
                    existing[k] = missing_key_default
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=keys, keep="last")
        else:
            combined = new_rows.copy()

        # Atomic write: readers never see a half-written file
        tmp_path = output_path.with_suffix(".csv.tmp")
        combined.to_csv(tmp_path, index=False, encoding="utf-8")
        os.replace(tmp_path, output_path)

    return str(output_path)