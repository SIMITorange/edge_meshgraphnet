"""
Module purpose:
    Lightweight logging utilities for console and file outputs.
Inputs:
    Functions accept message strings or dictionaries of metrics.
Outputs:
    Printed logs with timestamps and optional CSV/JSONL metric files.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any


def timestamp() -> str:
    """
    Returns a human-readable timestamp string.
    Inputs: None
    Outputs: Timestamp string in ISO-like format.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    """
    Print a message with timestamp.
    Inputs:
        msg: Message string.
    Outputs:
        None; message printed to stdout.
    """
    print(f"[{timestamp()}] {msg}")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append a JSON record as one line to a file.
    Inputs:
        path: Destination file path.
        record: Dictionary to serialize as JSON.
    Outputs:
        None; file is appended.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

