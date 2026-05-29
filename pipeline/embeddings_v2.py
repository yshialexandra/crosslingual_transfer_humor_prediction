"""
pipeline/embeddings_v2.py
=========================
Feature extraction for the v2 cross-lingual pipeline.

Text  : intfloat/multilingual-e5-large  (bidirectional, 1024-dim, 100 languages)
Audio : facebook/hubert-large-ls960-ft  (self-supervised acoustic encoder, 1024-dim)
"""


import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers.models.hubert.modeling_hubert import HubertModel
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor

log = logging.getLogger(__name__)

HUBERT_SR = 16_000
HUBERT_HIDDEN = 1024


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean_pool_with_mask(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _stride_pool(frames: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return frames
    T, H = frames.shape
    n_out = (T + stride - 1) // stride
    out = np.zeros((n_out, H), dtype=np.float32)
    for i in range(n_out):
        out[i] = frames[i * stride: i * stride + stride].mean(axis=0)
    return out


def _cache_path(cfg, suffix: str) -> Path:
    key = (
        f"{cfg.text_model_id}|"
        f"{cfg.audio_model_id}|"
        f"{cfg.audio_pool_stride}|"
        f"{cfg.max_text_tokens}|"
        f"{cfg.max_audio_tokens}"
    )
    key_hash = hashlib.md5(key.encode()).hexdigest()[:10]
    return cfg.artifact_dir / f"features_v2_{suffix}_{key_hash}.npz"


def _load_audio_file(path: str | Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != HUBERT_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=HUBERT_SR)
    return wav


def _row_keys(segments_df: pd.DataFrame) -> np.ndarray:
    return (
        segments_df[["lang", "title", "segment_id", "start", "end"]]
        .astype(str)
        .agg("|".join, axis=1)
        .to_numpy()
    )


# In embeddings_v2.py, replace _save_features with:
def _save_features(path: Path, arrays: List[np.ndarray], row_keys: np.ndarray) -> None:
    """
    Save as flat float32 array + offsets so mmap_mode works at load time.
    Object arrays can't be memory-mapped; concatenated flat arrays can.
    """
    lengths = np.array([a.shape[0] for a in arrays], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat    = np.concatenate(arrays, axis=0).astype(np.float32)  # (total_frames, H)
    np.savez(path, flat=flat, offsets=offsets, row_keys=row_keys)

# And replace _load_features with:
def _load_features_mmap(path: Path):
    """Returns mmap handles — no data loaded into RAM until accessed."""
    data    = np.load(path, allow_pickle=False, mmap_mode='r')
    return data['flat'], data['offsets'], data['row_keys']


# ---------------------------------------------------------------------------
# Text feature extraction
# ---------------------------------------------------------------------------

def _extract_text_features(
    texts: List[str],
    cfg,
    instruction: str,
    device: torch.device,
) -> List[np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_id)
    model = AutoModel.from_pretrained(cfg.text_model_id, dtype=torch.float16)
    model.eval().to(device)

    prefixed = [instruction + t for t in texts]
    all_embeddings: List[np.ndarray] = []

    for start in tqdm(range(0, len(prefixed), cfg.batch_size_embed_text), desc="e5-large text embeddings"):
        batch = prefixed[start: start + cfg.batch_size_embed_text]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=cfg.max_text_tokens,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        pooled = _mean_pool_with_mask(out.last_hidden_state, enc["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=-1)

        for vec in pooled.cpu().float().numpy():
            all_embeddings.append(vec[np.newaxis, :])   # (1, 1024)

    del model
    torch.cuda.empty_cache()
    return all_embeddings


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

def _extract_audio_features(
    rows: pd.DataFrame,
    cfg,
    device: torch.device,
) -> List[np.ndarray]:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.audio_model_id)
    model = HubertModel.from_pretrained(cfg.audio_model_id, dtype=torch.float16)
    model.eval().to(device)

    all_hidden: List[List[torch.Tensor]] = [[] for _ in range(len(rows))]
    pending_wavs: List[np.ndarray] = []
    pending_ids:  List[int] = []

    # FIX: bounded file cache — evict oldest entry when full so RAM doesn't
    # grow unboundedly across 912 video files.
    MAX_CACHED_FILES = 20
    file_cache: dict[str, np.ndarray] = {}

    _zero = np.zeros((1, HUBERT_HIDDEN), dtype=np.float32)

    def flush():
        if not pending_wavs:
            return
        inputs = feature_extractor(
            pending_wavs,
            sampling_rate=HUBERT_SR,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device, dtype=torch.float16)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model(input_values, attention_mask=attention_mask)

        for seg_idx, hidden in zip(pending_ids, out.last_hidden_state):
            all_hidden[seg_idx].append(hidden.detach().cpu().float())

        pending_wavs.clear()
        pending_ids.clear()
        torch.cuda.empty_cache()

    with torch.no_grad():
        for seg_idx, row in enumerate(
            tqdm(rows.itertuples(index=False), total=len(rows), desc="HuBERT audio embeddings")
        ):
            audio_path = (
                getattr(row, "foreground_path", None)
                or getattr(row, "audio_path", None)
            )
            start_sec = float(row.start)
            end_sec   = float(row.end)

            if (end_sec - start_sec) < 0.1 or not audio_path or not Path(str(audio_path)).exists():
                all_hidden[seg_idx].append(torch.from_numpy(_zero))
                continue

            try:
                if audio_path not in file_cache:
                    # Evict oldest entry if cache is full
                    if len(file_cache) >= MAX_CACHED_FILES:
                        file_cache.pop(next(iter(file_cache)))
                    file_cache[audio_path] = _load_audio_file(audio_path)

                wav = file_cache[audio_path]
                segment = wav[int(start_sec * HUBERT_SR): int(end_sec * HUBERT_SR)]

                if len(segment) < 400:
                    all_hidden[seg_idx].append(torch.from_numpy(_zero))
                    continue

                pending_wavs.append(segment)
                pending_ids.append(seg_idx)

                if len(pending_wavs) >= cfg.batch_size_embed_audio:
                    flush()

            except Exception as exc:
                log.warning("Audio extraction failed for %s: %s", audio_path, exc)
                all_hidden[seg_idx].append(torch.from_numpy(_zero))

    flush()

    result: List[np.ndarray] = []
    for seg_parts in all_hidden:
        if not seg_parts:
            result.append(_zero)
            continue
        frames = torch.cat(seg_parts, dim=0).numpy()
        frames = _stride_pool(frames, cfg.audio_pool_stride)
        frames = frames[: cfg.max_audio_tokens]
        result.append(frames.astype(np.float32))

    del model
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_or_extract_features_v2(
    segments_df: pd.DataFrame,
    cfg,
    text_instruction: str = "Represent this spoken comedy segment: ",
    force_recompute: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Load cached features or extract from scratch with e5-large + HuBERT-large.

    FIX: text and audio cache checks are now fully independent — a row-key
    mismatch on text no longer forces audio to recompute and re-download
    the model. Each modality is checked and saved separately.
    """
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    text_cache  = _cache_path(cfg, "text")
    audio_cache = _cache_path(cfg, "audio")
    device = torch.device(cfg.device)
    current_keys = _row_keys(segments_df)
    y = segments_df["label"].to_numpy(dtype=np.int64)

    # ── Text — fully independent of audio ────────────────────────────────────
    text_hidden = None
    if not force_recompute and text_cache.exists():
        try:
            arrays, cached_keys = _load_features(text_cache)
            if np.array_equal(cached_keys, current_keys):
                text_hidden = arrays
                print("Loaded text features from cache:", text_cache)
            else:
                print("Text cache row-key mismatch — will recompute text only.")
        except Exception as e:
            print(f"Text cache load failed ({e}) — recomputing.")

    if text_hidden is None:
        print(f"Extracting text features with {cfg.text_model_id} ...")
        texts = segments_df["text"].fillna("").tolist()
        text_hidden = _extract_text_features(texts, cfg, text_instruction, device)
        _save_features(text_cache, text_hidden, current_keys)
        print("Saved text features ->", text_cache)

    # ── Audio — fully independent of text ────────────────────────────────────
    audio_hidden = None
    if not force_recompute and audio_cache.exists():
        try:
            arrays, cached_keys = _load_features(audio_cache)
            if np.array_equal(cached_keys, current_keys):
                audio_hidden = arrays
                print("Loaded audio features from cache:", audio_cache)
            else:
                print("Audio cache row-key mismatch — will recompute audio only.")
        except Exception as e:
            print(f"Audio cache load failed ({e}) — recomputing.")

    if audio_hidden is None:
        print(f"Extracting audio features with {cfg.audio_model_id} ...")
        audio_hidden = _extract_audio_features(segments_df, cfg, device)
        _save_features(audio_cache, audio_hidden, current_keys)
        print("Saved audio features ->", audio_cache)

    assert len(text_hidden) == len(y), \
        f"text_hidden {len(text_hidden)} != labels {len(y)}"
    assert len(audio_hidden) == len(y), \
        f"audio_hidden {len(audio_hidden)} != labels {len(y)}"

    """
pipeline/embeddings_v2.py
=========================
Feature extraction for the v2 cross-lingual pipeline.

Text  : intfloat/multilingual-e5-large  (bidirectional, 1024-dim, 100 languages)
Audio : facebook/hubert-large-ls960-ft  (self-supervised acoustic encoder, 1024-dim)
"""


import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers.models.hubert.modeling_hubert import HubertModel
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor

log = logging.getLogger(__name__)

HUBERT_SR = 16_000
HUBERT_HIDDEN = 1024


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean_pool_with_mask(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _stride_pool(frames: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return frames
    T, H = frames.shape
    n_out = (T + stride - 1) // stride
    out = np.zeros((n_out, H), dtype=np.float32)
    for i in range(n_out):
        out[i] = frames[i * stride: i * stride + stride].mean(axis=0)
    return out


def _cache_path(cfg, suffix: str) -> Path:
    key = (
        f"{cfg.text_model_id}|"
        f"{cfg.audio_model_id}|"
        f"{cfg.audio_pool_stride}|"
        f"{cfg.max_text_tokens}|"
        f"{cfg.max_audio_tokens}"
    )
    key_hash = hashlib.md5(key.encode()).hexdigest()[:10]
    return cfg.artifact_dir / f"features_v2_{suffix}_{key_hash}.npz"


def _load_audio_file(path: str | Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != HUBERT_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=HUBERT_SR)
    return wav


def _row_keys(segments_df: pd.DataFrame) -> np.ndarray:
    return (
        segments_df[["lang", "title", "segment_id", "start", "end"]]
        .astype(str)
        .agg("|".join, axis=1)
        .to_numpy()
    )


def _save_features(path: Path, arrays: List[np.ndarray], row_keys: np.ndarray) -> None:
    """
    Save features segment-by-segment into a single uncompressed npz.

    FIX: np.savez_compressed on 85k object arrays spikes RAM massively
    during compression and kills the kernel. np.savez (uncompressed) writes
    directly to disk with no in-memory compression buffer.
    The file will be larger (~2-4x) but saves will complete reliably.
    """
    np.savez(
        path,
        hidden=np.array(arrays, dtype=object),
        row_keys=row_keys,
    )


def _load_features(path: Path) -> List[np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return list(data["hidden"]), data["row_keys"]


# ---------------------------------------------------------------------------
# Text feature extraction
# ---------------------------------------------------------------------------

def _extract_text_features(
    texts: List[str],
    cfg,
    instruction: str,
    device: torch.device,
) -> List[np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_id)
    model = AutoModel.from_pretrained(cfg.text_model_id, dtype=torch.float16)
    model.eval().to(device)

    prefixed = [instruction + t for t in texts]
    all_embeddings: List[np.ndarray] = []

    for start in tqdm(range(0, len(prefixed), cfg.batch_size_embed_text), desc="e5-large text embeddings"):
        batch = prefixed[start: start + cfg.batch_size_embed_text]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=cfg.max_text_tokens,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        pooled = _mean_pool_with_mask(out.last_hidden_state, enc["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=-1)

        for vec in pooled.cpu().float().numpy():
            all_embeddings.append(vec[np.newaxis, :])   # (1, 1024)

    del model
    torch.cuda.empty_cache()
    return all_embeddings


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

def _extract_audio_features(
    rows: pd.DataFrame,
    cfg,
    device: torch.device,
) -> List[np.ndarray]:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.audio_model_id)
    model = HubertModel.from_pretrained(cfg.audio_model_id, dtype=torch.float16)
    model.eval().to(device)

    all_hidden: List[List[torch.Tensor]] = [[] for _ in range(len(rows))]
    pending_wavs: List[np.ndarray] = []
    pending_ids:  List[int] = []

    # FIX: bounded file cache — evict oldest entry when full so RAM doesn't
    # grow unboundedly across 912 video files.
    MAX_CACHED_FILES = 20
    file_cache: dict[str, np.ndarray] = {}

    _zero = np.zeros((1, HUBERT_HIDDEN), dtype=np.float32)

    def flush():
        if not pending_wavs:
            return
        inputs = feature_extractor(
            pending_wavs,
            sampling_rate=HUBERT_SR,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device, dtype=torch.float16)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model(input_values, attention_mask=attention_mask)

        for seg_idx, hidden in zip(pending_ids, out.last_hidden_state):
            all_hidden[seg_idx].append(hidden.detach().cpu().float())

        pending_wavs.clear()
        pending_ids.clear()
        torch.cuda.empty_cache()

    with torch.no_grad():
        for seg_idx, row in enumerate(
            tqdm(rows.itertuples(index=False), total=len(rows), desc="HuBERT audio embeddings")
        ):
            audio_path = (
                getattr(row, "foreground_path", None)
                or getattr(row, "audio_path", None)
            )
            start_sec = float(row.start)
            end_sec   = float(row.end)

            if (end_sec - start_sec) < 0.1 or not audio_path or not Path(str(audio_path)).exists():
                all_hidden[seg_idx].append(torch.from_numpy(_zero))
                continue

            try:
                if audio_path not in file_cache:
                    # Evict oldest entry if cache is full
                    if len(file_cache) >= MAX_CACHED_FILES:
                        file_cache.pop(next(iter(file_cache)))
                    file_cache[audio_path] = _load_audio_file(audio_path)

                wav = file_cache[audio_path]
                segment = wav[int(start_sec * HUBERT_SR): int(end_sec * HUBERT_SR)]

                if len(segment) < 400:
                    all_hidden[seg_idx].append(torch.from_numpy(_zero))
                    continue

                pending_wavs.append(segment)
                pending_ids.append(seg_idx)

                if len(pending_wavs) >= cfg.batch_size_embed_audio:
                    flush()

            except Exception as exc:
                log.warning("Audio extraction failed for %s: %s", audio_path, exc)
                all_hidden[seg_idx].append(torch.from_numpy(_zero))

    flush()

    result: List[np.ndarray] = []
    for seg_parts in all_hidden:
        if not seg_parts:
            result.append(_zero)
            continue
        frames = torch.cat(seg_parts, dim=0).numpy()
        frames = _stride_pool(frames, cfg.audio_pool_stride)
        frames = frames[: cfg.max_audio_tokens]
        result.append(frames.astype(np.float32))

    del model
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_or_extract_features_v2(
    segments_df: pd.DataFrame,
    cfg,
    text_instruction: str = "Represent this spoken comedy segment: ",
    force_recompute: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Load cached features or extract from scratch with e5-large + HuBERT-large.

    FIX: text and audio cache checks are now fully independent — a row-key
    mismatch on text no longer forces audio to recompute and re-download
    the model. Each modality is checked and saved separately.
    """
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    text_cache  = _cache_path(cfg, "text")
    audio_cache = _cache_path(cfg, "audio")
    device = torch.device(cfg.device)
    current_keys = _row_keys(segments_df)
    y = segments_df["label"].to_numpy(dtype=np.int64)

    # ── Text — fully independent of audio ────────────────────────────────────
    text_hidden = None
    if not force_recompute and text_cache.exists():
        try:
            arrays, cached_keys = _load_features(text_cache)
            if np.array_equal(cached_keys, current_keys):
                text_hidden = arrays
                print("Loaded text features from cache:", text_cache)
            else:
                print("Text cache row-key mismatch — will recompute text only.")
        except Exception as e:
            print(f"Text cache load failed ({e}) — recomputing.")

    if text_hidden is None:
        print(f"Extracting text features with {cfg.text_model_id} ...")
        texts = segments_df["text"].fillna("").tolist()
        text_hidden = _extract_text_features(texts, cfg, text_instruction, device)
        _save_features(text_cache, text_hidden, current_keys)
        print("Saved text features ->", text_cache)

    # ── Audio — fully independent of text ────────────────────────────────────
    audio_hidden = None
    if not force_recompute and audio_cache.exists():
        try:
            arrays, cached_keys = _load_features(audio_cache)
            if np.array_equal(cached_keys, current_keys):
                audio_hidden = arrays
                print("Loaded audio features from cache:", audio_cache)
            else:
                print("Audio cache row-key mismatch — will recompute audio only.")
        except Exception as e:
            print(f"Audio cache load failed ({e}) — recomputing.")

    if audio_hidden is None:
        print(f"Extracting audio features with {cfg.audio_model_id} ...")
        audio_hidden = _extract_audio_features(segments_df, cfg, device)
        _save_features(audio_cache, audio_hidden, current_keys)
        print("Saved audio features ->", audio_cache)

    assert len(text_hidden) == len(y), \
        f"text_hidden {len(text_hidden)} != labels {len(y)}"
    assert len(audio_hidden) == len(y), \
        f"audio_hidden {len(audio_hidden)} != labels {len(y)}"

    return text_hidden, audio_hidden, y