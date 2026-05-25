from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class PipelineConfig:
    project_root: Path = Path.cwd()
    lang: str = "es"
    audio_root_name: str = "es-test"
    sample_limit: int | None = 8
    random_seed: int = 42

    qwen_model_id: str = "Qwen/Qwen3-0.6B"
    whisper_model_id: str = "openai/whisper-large-v3-turbo"

    target_sr: int = 16000
    max_audio_window_sec: float = 30.0
    audio_pool_stride: int = 8
    max_audio_tokens: int = 256
    max_text_tokens: int = 256

    pre_laughter_window_sec: float = 2.0
    min_overlap_sec: float = 0.05

    crowd_frame_sec: float = 0.10
    crowd_hop_sec: float = 0.05
    crowd_threshold_quantile: float = 0.78
    crowd_min_interval_sec: float = 0.25
    crowd_merge_gap_sec: float = 0.35

    batch_size_embed_text: int = 4
    batch_size_train: int = 16
    epochs: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-3
    fusion_hidden_dim: int = 512
    fusion_num_heads: int = 8
    dropout: float = 0.25

    @property
    def transcript_dir(self) -> Path:
        return self.project_root / "asr-output" / self.lang

    @property
    def audio_root(self) -> Path:
        return self.project_root / self.audio_root_name

    @property
    def artifact_dir(self) -> Path:
        path = self.project_root / "pipeline_artifacts"
        path.mkdir(exist_ok=True)
        return path

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def dtype(self):
        return torch.float16 if self.device == "cuda" else torch.float32
