from __future__ import annotations

import json
from pathlib import Path
from difflib import SequenceMatcher

import librosa
import numpy as np

from .config import PipelineConfig
from .data import normalize_title


def merge_intervals(intervals: list[tuple[float, float]], merge_gap_sec: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(s), float(e)) for s, e in merged]


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


class RMSLaughterDetector:

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def detect(self, row) -> list[tuple[float, float]]:
        crowd_path = Path(row.crowd_path)
        y, sr = librosa.load(crowd_path, sr=self.cfg.target_sr, mono=True)
        if len(y) == 0:
            return []

        frame_length = max(1, int(self.cfg.crowd_frame_sec * sr))
        hop_length = max(1, int(self.cfg.crowd_hop_sec * sr))
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        threshold = np.quantile(rms, self.cfg.crowd_threshold_quantile)
        active = rms >= threshold

        intervals = []
        start = None
        for t, is_active in zip(times, active):
            if is_active and start is None:
                start = float(t)
            elif not is_active and start is not None:
                end = float(t)
                if end - start >= self.cfg.crowd_min_interval_sec:
                    intervals.append((start, end))
                start = None
        if start is not None:
            end = float(times[-1] + self.cfg.crowd_frame_sec)
            if end - start >= self.cfg.crowd_min_interval_sec:
                intervals.append((start, end))

        return merge_intervals(intervals, self.cfg.crowd_merge_gap_sec)

    def label_segment(self, start: float, end: float, laughter_intervals: list[tuple[float, float]]) -> int:
        for laugh_start, laugh_end in laughter_intervals:
            overlaps_laughter = interval_overlap(start, end, laugh_start, laugh_end) >= self.cfg.min_overlap_sec
            precedes_laughter = (laugh_start - self.cfg.pre_laughter_window_sec) <= end <= laugh_start
            if overlaps_laughter or precedes_laughter:
                return 1
        return 0

    def activity_score(self, start: float, end: float, laughter_intervals: list[tuple[float, float]]) -> float:
        duration = max(end - start, 1e-6)
        overlap = sum(interval_overlap(start, end, s, e) for s, e in laughter_intervals)
        near = any((s - self.cfg.pre_laughter_window_sec) <= end <= s for s, _ in laughter_intervals)
        return min(1.0, overlap / duration) + (0.5 if near else 0.0)


class AnnotationLaughterDetector(RMSLaughterDetector):

    def __init__(self, cfg: PipelineConfig, annotation_root: Path | None = None, min_match_score: float = 0.50):
        super().__init__(cfg)
        self.annotation_root = annotation_root or (cfg.project_root / "laughter_crowd")
        self.min_match_score = min_match_score
        self._files_by_lang: dict[str, list[Path]] = {}
        self._match_cache: dict[tuple[str, str], Path | None] = {}

    @staticmethod
    def _strip_annotation_suffix(stem: str) -> str:
        for suffix in (" Crowd", " Foreground", " Crowd(1)", " Foreground(1)"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _annotation_files(self, lang: str) -> list[Path]:
        if lang not in self._files_by_lang:
            lang_dir = self.annotation_root / lang
            self._files_by_lang[lang] = sorted(lang_dir.glob("*.json")) if lang_dir.exists() else []
        return self._files_by_lang[lang]

    def find_annotation_file(self, row) -> Path | None:
        lang = str(row.lang)
        cache_key = (lang, str(row.audio_title))
        if cache_key in self._match_cache:
            return self._match_cache[cache_key]

        files = self._annotation_files(lang)
        if not files:
            self._match_cache[cache_key] = None
            return None

        targets = [
            normalize_title(str(row.audio_title)),
            normalize_title(Path(row.json_path).stem),
            normalize_title(Path(row.foreground_path).stem),
        ]

        best_file, best_score = None, -1.0
        for path in files:
            candidate = normalize_title(self._strip_annotation_suffix(path.stem))
            score = max(SequenceMatcher(None, target, candidate).ratio() for target in targets)
            if score > best_score:
                best_file, best_score = path, score

        if best_score < self.min_match_score:
            best_file = None
        self._match_cache[cache_key] = best_file
        return best_file

    def detect(self, row) -> list[tuple[float, float]]:
        annotation_path = self.find_annotation_file(row)
        if annotation_path is None:
            return []

        data = json.loads(annotation_path.read_text(encoding="utf-8", errors="replace"))
        entries = data.values() if isinstance(data, dict) else data
        intervals = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "start_sec" not in entry or "end_sec" not in entry:
                continue
            start = float(entry["start_sec"])
            end = float(entry["end_sec"])
            if end > start:
                intervals.append((start, end))
        return merge_intervals(intervals, self.cfg.crowd_merge_gap_sec)
