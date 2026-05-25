from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .config import PipelineConfig


def normalize_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    replacements = {
        "｜": " ",
        "|": " ",
        "：": " ",
        ":": " ",
        "´¢£": " ",
        "ÔÇô": " ",
        "´╝Ü": " ",
        "├¡": "i",
        "├®": "e",
        "├í": "a",
        "├▒": "n",
        "├║": "u",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = s.lower()
    s = re.sub(r"[^a-z0-9áéíóúüñç]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_audio_role_suffix(stem: str) -> tuple[str, str | None]:
    for role in ("Foreground", "Crowd"):
        suffix = f" {role}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], role.lower()
    return stem, None


def collect_audio_items(audio_root: Path) -> list[dict]:
    """Collect one audio item per video.

    Supports both layouts:
    - nested: <title>/Foreground.wav and optional <title>/Crowd.wav
    - flat: <title> Foreground.wav and optional <title> Crowd.wav

    Flat Foreground/Crowd files are grouped into one item by title so a video is
    not counted twice.
    """
    items = []
    flat_groups: dict[str, dict] = {}
    for p in audio_root.iterdir():
        if p.is_dir() and (p / "Foreground.wav").exists():
            items.append(
                {
                    "match_key": p.name,
                    "audio_dir": p,
                    "audio_title": p.name,
                    "foreground_path": p / "Foreground.wav",
                    "crowd_path": p / "Crowd.wav" if (p / "Crowd.wav").exists() else p / "Foreground.wav",
                }
            )
        elif p.is_file() and p.suffix.lower() == ".wav":
            title, role = strip_audio_role_suffix(p.stem)
            group = flat_groups.setdefault(
                title,
                {
                    "match_key": title,
                    "audio_dir": p.parent,
                    "audio_title": title,
                    "foreground_path": None,
                    "crowd_path": None,
                    "single_path": None,
                },
            )
            if role == "foreground":
                group["foreground_path"] = p
            elif role == "crowd":
                group["crowd_path"] = p
            else:
                group["single_path"] = p

    for group in flat_groups.values():
        if group["foreground_path"] is None:
            group["foreground_path"] = group["single_path"] or group["crowd_path"]
        if group["crowd_path"] is None:
            group["crowd_path"] = group["single_path"] or group["foreground_path"]
        group.pop("single_path", None)
        if group["foreground_path"] is not None:
            items.append(group)
    return items


def discover_matched_samples(
    transcript_dir: Path,
    audio_root: Path,
    sample_limit: int | None = None,
    lang: str | None = None,
) -> pd.DataFrame:
    audio_items = collect_audio_items(audio_root)
    candidate_pairs = []
    for json_path in sorted(transcript_dir.glob("*.json")):
        target = normalize_title(json_path.stem)
        for audio_idx, item in enumerate(audio_items):
            score = SequenceMatcher(None, target, normalize_title(item["match_key"])).ratio()
            candidate_pairs.append((score, json_path, audio_idx))

    rows = []
    used_jsons = set()
    used_audio = set()
    for score, json_path, audio_idx in sorted(candidate_pairs, reverse=True, key=lambda x: x[0]):
        if score < 0.55 or json_path in used_jsons or audio_idx in used_audio:
            continue
        item = audio_items[audio_idx]
        rows.append(
            {
                "json_path": json_path,
                "audio_dir": item["audio_dir"],
                "foreground_path": item["foreground_path"],
                "crowd_path": item["crowd_path"],
                "match_score": score,
                "json_title": json_path.stem,
                "audio_title": item["audio_title"],
                "lang": lang,
            }
        )
        used_jsons.add(json_path)
        used_audio.add(audio_idx)
    df = pd.DataFrame(rows).sort_values("match_score", ascending=False).reset_index(drop=True)
    if sample_limit is not None:
        df = df.head(sample_limit).copy()
    return df


def load_transcript_segments(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    segments = []
    for i, chunk in enumerate(data.get("chunks", [])):
        text = str(chunk.get("text", "")).strip()
        if "timestamp" in chunk and chunk["timestamp"] is not None:
            start, end = chunk["timestamp"]
        else:
            start = chunk.get("start", 0.0)
            end = chunk.get("end", start)

        start = float(start)
        end = float(end)
        if not text or end <= start:
            continue
        segments.append({"segment_id": i, "text": text, "start": start, "end": end})
    return segments


def build_segment_dataframe(
    matched_df: pd.DataFrame,
    detector,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    rows = []
    for sample_idx, row in tqdm(matched_df.iterrows(), total=len(matched_df), desc="building segments"):
        segments = load_transcript_segments(row.json_path)
        laughter_intervals = detector.detect(row)
        for seg in segments:
            label = detector.label_segment(seg["start"], seg["end"], laughter_intervals)
            rows.append(
                {
                    "lang": row.lang if "lang" in row else cfg.lang,
                    "sample_idx": sample_idx,
                    "segment_id": seg["segment_id"],
                    "title": row.audio_title,
                    "json_path": str(row.json_path),
                    "foreground_path": str(row.foreground_path),
                    "crowd_path": str(row.crowd_path),
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["end"] - seg["start"],
                    "text": seg["text"],
                    "label": label,
                    "crowd_score": detector.activity_score(seg["start"], seg["end"], laughter_intervals),
                    "num_laughter_intervals_in_video": len(laughter_intervals),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No segments were built. Check transcript/audio matching or transcript chunk format.")
    return df


def split_by_video(df: pd.DataFrame, seed: int):
    titles = df["title"].drop_duplicates().tolist()
    if len(titles) < 3:
        idx = pd.Series(range(len(df))).to_numpy()
        stratify = df["label"] if df["label"].nunique() > 1 else None
        train_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=stratify)
        tmp_labels = df.iloc[tmp_idx]["label"]
        stratify_tmp = tmp_labels if tmp_labels.nunique() > 1 else None
        val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, random_state=seed, stratify=stratify_tmp)
        split = pd.Series("train", index=df.index)
        split.iloc[val_idx] = "val"
        split.iloc[test_idx] = "test"
        return split

    train_titles, tmp_titles = train_test_split(titles, test_size=0.30, random_state=seed)
    val_titles, test_titles = train_test_split(tmp_titles, test_size=0.50, random_state=seed)
    split = pd.Series("train", index=df.index)
    split[df["title"].isin(val_titles)] = "val"
    split[df["title"].isin(test_titles)] = "test"
    return split
