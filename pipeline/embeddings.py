from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer

from .config import PipelineConfig


def load_qwen_text_encoder(cfg: PipelineConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_model_id,
        torch_dtype=cfg.dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(cfg.device)
    model.eval()
    return tokenizer, model


def embed_texts_with_qwen(texts: list[str], cfg: PipelineConfig) -> list[np.ndarray]:
    tokenizer, model = load_qwen_text_encoder(cfg)
    all_hidden = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), cfg.batch_size_embed_text), desc="Qwen3 text sequence embeddings"):
            batch = texts[i : i + cfg.batch_size_embed_text]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_text_tokens,
                return_tensors="pt",
            ).to(cfg.device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            last_hidden = outputs.hidden_states[-1].detach().cpu().float().numpy()
            attention_mask = inputs["attention_mask"].detach().cpu().numpy().astype(bool)
            for hidden_i, mask_i in zip(last_hidden, attention_mask):
                all_hidden.append(hidden_i[mask_i])
    del model
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return all_hidden


def load_whisper_audio_encoder(cfg: PipelineConfig):
    processor = AutoProcessor.from_pretrained(cfg.whisper_model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        cfg.whisper_model_id,
        torch_dtype=cfg.dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(cfg.device)
    encoder = model.get_encoder()
    encoder.eval()
    return processor, encoder


def audio_windows(y: np.ndarray, sr: int, start: float, end: float, cfg: PipelineConfig) -> list[np.ndarray]:
    start_i = max(0, int(start * sr))
    end_i = min(len(y), int(end * sr))
    segment = y[start_i:end_i]
    if len(segment) == 0:
        segment = np.zeros(int(0.25 * sr), dtype=np.float32)

    max_len = int(cfg.max_audio_window_sec * sr)
    if len(segment) <= max_len:
        return [segment.astype(np.float32)]

    windows = []
    for left in range(0, len(segment), max_len):
        chunk = segment[left : left + max_len]
        if len(chunk) > int(0.10 * sr):
            windows.append(chunk.astype(np.float32))
    return windows


def temporal_mean_pool(hidden: torch.Tensor, cfg: PipelineConfig) -> torch.Tensor:
    stride = cfg.audio_pool_stride
    if stride <= 1 or hidden.shape[0] <= stride:
        return hidden
    usable = (hidden.shape[0] // stride) * stride
    pooled = hidden[:usable].reshape(-1, stride, hidden.shape[-1]).mean(dim=1)
    if usable < hidden.shape[0]:
        pooled = torch.cat([pooled, hidden[usable:].mean(dim=0, keepdim=True)], dim=0)
    return pooled


def limit_audio_tokens(hidden: torch.Tensor, cfg: PipelineConfig) -> torch.Tensor:
    if hidden.shape[0] <= cfg.max_audio_tokens:
        return hidden
    idx = torch.linspace(0, hidden.shape[0] - 1, steps=cfg.max_audio_tokens).long()
    return hidden[idx]


def embed_audio_segments_with_whisper(df: pd.DataFrame, cfg: PipelineConfig) -> list[np.ndarray]:
    processor, encoder = load_whisper_audio_encoder(cfg)
    cache: dict[str, tuple[np.ndarray, int]] = {}
    all_hidden = []
    with torch.no_grad():
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Whisper audio sequence embeddings"):
            path = row.foreground_path
            if path not in cache:
                y, sr = librosa.load(path, sr=cfg.target_sr, mono=True)
                cache[path] = (y, sr)
            y, sr = cache[path]
            windows = audio_windows(y, sr, float(row.start), float(row.end), cfg)
            inputs = processor(windows, sampling_rate=sr, return_tensors="pt").to(cfg.device)
            input_features = inputs.input_features.to(dtype=cfg.dtype)
            outputs = encoder(input_features=input_features)
            window_hidden = []
            for hidden in outputs.last_hidden_state:
                pooled = temporal_mean_pool(hidden.detach().cpu().float(), cfg)
                window_hidden.append(pooled)
            hidden_seq = torch.cat(window_hidden, dim=0)
            hidden_seq = limit_audio_tokens(hidden_seq, cfg)
            all_hidden.append(hidden_seq.numpy())
    del encoder
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return all_hidden


def load_or_extract_features(segments_df: pd.DataFrame, cfg: PipelineConfig, force_recompute: bool = False):
    features_npz = cfg.artifact_dir / "crosslingual_sequence_features_qwen3_whisper.npz"
    row_keys = segments_df[["lang", "title", "segment_id", "start", "end"]].astype(str).agg("|".join, axis=1).to_numpy()
    use_cache = False
    if features_npz.exists() and not force_recompute:
        data = np.load(features_npz, allow_pickle=True)
        cached_labels = np.asarray(data["labels"], dtype=np.int64)
        cached_row_keys = data["row_keys"] if "row_keys" in data.files else None
        cache_matches = (
            len(cached_labels) == len(segments_df)
            and cached_row_keys is not None
            and np.array_equal(cached_row_keys, row_keys)
        )
        if cache_matches:
            text_hidden = list(data["text_hidden"])
            audio_hidden = list(data["audio_hidden"])
            y = cached_labels
            use_cache = True
            print("loaded:", features_npz)
        else:
            print(
                "feature cache mismatch; recomputing features:",
                f"cache={len(cached_labels)} current_segments={len(segments_df)}",
            )

    if not use_cache:
        text_hidden = embed_texts_with_qwen(segments_df["text"].tolist(), cfg)
        audio_hidden = embed_audio_segments_with_whisper(segments_df, cfg)
        y = segments_df["label"].to_numpy(dtype=np.int64)
        np.savez_compressed(
            features_npz,
            text_hidden=np.array(text_hidden, dtype=object),
            audio_hidden=np.array(audio_hidden, dtype=object),
            labels=y,
            row_keys=row_keys,
        )
        print("saved:", features_npz)

    return text_hidden, audio_hidden, y
