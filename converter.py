import json
import os
from datetime import datetime
from pathlib import Path

from prompts import FINE_TUNE_SYSTEM_PROMPT


# ── Save ───────────────────────────────────────────────────────────────────────
def save_results(conversations: list[dict], output_dir: str) -> dict:
    """Persist *conversations* to three files inside *output_dir*.

    Returns a dict with keys: json, jsonl, log, count.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path  = out / "conversations.json"
    jsonl_path = out / "train.jsonl"
    log_path   = out / "generacios_naplo.txt"

    # 1. Full JSON dump (human-readable)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(conversations, fh, ensure_ascii=False, indent=2)

    # 2. JSONL for fine-tuning
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for conv in conversations:
            record = {
                "messages": [
                    {"role": "system",    "content": FINE_TUNE_SYSTEM_PROMPT},
                    {"role": "user",      "content": conv.get("user_message", "")},
                    {"role": "assistant", "content": conv.get("assistant_message", "")},
                ]
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 3. Human-readable log
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"Generálás dátuma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Összes párbeszéd: {len(conversations)}\n")
        fh.write("─" * 60 + "\n")
        for i, conv in enumerate(conversations, 1):
            tema = conv.get("tema", "?")
            fh.write(f"{i:4d}. {tema}\n")

    return {
        "json":  str(json_path),
        "jsonl": str(jsonl_path),
        "log":   str(log_path),
        "count": len(conversations),
    }


# ── Load existing (for resume support) ────────────────────────────────────────
def load_existing(output_dir: str) -> list[dict]:
    """Load previously saved conversations from *output_dir*, or return []."""
    json_path = Path(output_dir) / "conversations.json"
    if not json_path.exists():
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ── Statistics ─────────────────────────────────────────────────────────────────
def get_stats(conversations: list[dict]) -> dict:
    """Return basic statistics about *conversations*."""
    count = len(conversations)
    if count == 0:
        return {
            "count":            0,
            "unique_topics":    0,
            "avg_user_len":     0,
            "avg_assistant_len": 0,
        }

    topics = {c.get("tema", "") for c in conversations}
    avg_user = sum(len(c.get("user_message", "")) for c in conversations) / count
    avg_asst = sum(len(c.get("assistant_message", "")) for c in conversations) / count

    return {
        "count":             count,
        "unique_topics":     len(topics),
        "avg_user_len":      round(avg_user, 1),
        "avg_assistant_len": round(avg_asst, 1),
    }
