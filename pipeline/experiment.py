from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .data import build_segment_dataframe, discover_matched_samples


def build_language_dataset(
    lang: str,
    audio_root: Path,
    detector,
    cfg: PipelineConfig,
    sample_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transcript_dir = cfg.project_root / "asr-output" / lang
    matched = discover_matched_samples(transcript_dir, audio_root, sample_limit=sample_limit, lang=lang)
    segments = build_segment_dataframe(matched, detector, cfg)
    segments["lang"] = lang
    return matched, segments


def build_crosslingual_dataset(
    langs: list[str],
    audio_roots: dict[str, Path],
    detector,
    cfg: PipelineConfig,
    sample_limits: dict[str, int | None] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    matched_by_lang = {}
    segment_frames = []
    sample_limits = sample_limits or {}
    for lang in langs:
        matched, segments = build_language_dataset(
            lang,
            audio_roots[lang],
            detector,
            cfg,
            sample_limit=sample_limits.get(lang, cfg.sample_limit),
        )
        matched_by_lang[lang] = matched
        segment_frames.append(segments)
    all_segments = pd.concat(segment_frames, ignore_index=True)
    return matched_by_lang, all_segments
