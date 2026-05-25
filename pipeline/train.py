from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import PipelineConfig
from .models import MultimodalSegmentClassifier


class SegmentFeatureDataset(Dataset):
    def __init__(self, text_hidden, audio_hidden, labels, indices):
        self.text_hidden = [self._to_float_tensor(text_hidden[i]) for i in indices]
        self.audio_hidden = [self._to_float_tensor(audio_hidden[i]) for i in indices]
        self.labels = torch.tensor(labels[indices], dtype=torch.long)

    @staticmethod
    def _to_float_tensor(x):
        return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text_hidden": self.text_hidden[idx],
            "audio_hidden": self.audio_hidden[idx],
            "labels": self.labels[idx],
        }


def multimodal_collate_fn(batch):
    text_lengths = torch.tensor([item["text_hidden"].shape[0] for item in batch], dtype=torch.long)
    audio_lengths = torch.tensor([item["audio_hidden"].shape[0] for item in batch], dtype=torch.long)
    labels = torch.stack([item["labels"] for item in batch])

    max_text_len = int(text_lengths.max())
    max_audio_len = int(audio_lengths.max())
    text_dim = batch[0]["text_hidden"].shape[-1]
    audio_dim = batch[0]["audio_hidden"].shape[-1]

    text_batch = torch.zeros(len(batch), max_text_len, text_dim, dtype=torch.float32)
    audio_batch = torch.zeros(len(batch), max_audio_len, audio_dim, dtype=torch.float32)
    text_mask = torch.zeros(len(batch), max_text_len, dtype=torch.bool)
    audio_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        t_len = item["text_hidden"].shape[0]
        a_len = item["audio_hidden"].shape[0]
        text_batch[i, :t_len] = item["text_hidden"]
        audio_batch[i, :a_len] = item["audio_hidden"]
        text_mask[i, :t_len] = True
        audio_mask[i, :a_len] = True

    return {
        "text_hidden": text_batch,
        "audio_hidden": audio_batch,
        "text_mask": text_mask,
        "audio_mask": audio_mask,
        "labels": labels,
    }


def make_loader(text_hidden, audio_hidden, labels, indices, cfg: PipelineConfig, shuffle: bool = False):
    ds = SegmentFeatureDataset(text_hidden, audio_hidden, labels, indices)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size_train,
        shuffle=shuffle,
        collate_fn=multimodal_collate_fn,
    )


def make_loaders(text_hidden, audio_hidden, labels, split_series, cfg: PipelineConfig):
    idx = np.arange(len(labels))
    split_values = split_series.to_numpy()
    train_idx = idx[split_values == "train"]
    val_idx = idx[split_values == "val"]
    test_idx = idx[split_values == "test"]

    loaders = {}
    for name, indices, shuffle in [("train", train_idx, True), ("val", val_idx, False), ("test", test_idx, False)]:
        loaders[name] = make_loader(text_hidden, audio_hidden, labels, indices, cfg, shuffle=shuffle)
    return loaders


def evaluate_model(model, loader, cfg: PipelineConfig):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            text_b = batch["text_hidden"].to(cfg.device)
            audio_b = batch["audio_hidden"].to(cfg.device)
            text_mask = batch["text_mask"].to(cfg.device)
            audio_mask = batch["audio_mask"].to(cfg.device)
            labels_b = batch["labels"].to(cfg.device)
            logits = model(text_b, audio_b, text_mask, audio_mask)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels_b.cpu().numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[1], average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_1": precision,
        "recall_1": recall,
        "f1_1": f1,
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
    }


def train_one_model(mode: str, text_hidden, audio_hidden, labels, split_series, cfg: PipelineConfig):
    loaders = make_loaders(text_hidden, audio_hidden, labels, split_series, cfg)
    text_dim = text_hidden[0].shape[-1]
    audio_dim = audio_hidden[0].shape[-1]
    model = MultimodalSegmentClassifier(
        text_dim,
        audio_dim,
        hidden_dim=cfg.fusion_hidden_dim,
        num_heads=cfg.fusion_num_heads,
        dropout=cfg.dropout,
        mode=mode,
    ).to(cfg.device)

    train_labels = labels[split_series.to_numpy() == "train"]
    counts = np.bincount(train_labels, minlength=2)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=cfg.device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            text_b = batch["text_hidden"].to(cfg.device)
            audio_b = batch["audio_hidden"].to(cfg.device)
            text_mask = batch["text_mask"].to(cfg.device)
            audio_mask = batch["audio_mask"].to(cfg.device)
            labels_b = batch["labels"].to(cfg.device)
            logits = model(text_b, audio_b, text_mask, audio_mask)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_model(model, loaders["val"], cfg)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **{k: v for k, v in val_metrics.items() if not isinstance(v, np.ndarray)},
        }
        history.append(row)
        print(f"{mode:>15} epoch {epoch:02d} loss={row['loss']:.4f} val_f1={row['f1_1']:.4f}")

        if val_metrics["f1_1"] > best_val_f1:
            best_val_f1 = val_metrics["f1_1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, loaders["test"], cfg)
    return model, pd.DataFrame(history), test_metrics


def train_transfer_model(
    mode: str,
    text_hidden,
    audio_hidden,
    labels,
    train_idx,
    val_idx,
    cfg: PipelineConfig,
):
    train_loader = make_loader(text_hidden, audio_hidden, labels, train_idx, cfg, shuffle=True)
    val_loader = make_loader(text_hidden, audio_hidden, labels, val_idx, cfg, shuffle=False)
    text_dim = text_hidden[0].shape[-1]
    audio_dim = audio_hidden[0].shape[-1]
    model = MultimodalSegmentClassifier(
        text_dim,
        audio_dim,
        hidden_dim=cfg.fusion_hidden_dim,
        num_heads=cfg.fusion_num_heads,
        dropout=cfg.dropout,
        mode=mode,
    ).to(cfg.device)

    train_labels = labels[train_idx]
    counts = np.bincount(train_labels, minlength=2)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=cfg.device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            text_b = batch["text_hidden"].to(cfg.device)
            audio_b = batch["audio_hidden"].to(cfg.device)
            text_mask = batch["text_mask"].to(cfg.device)
            audio_mask = batch["audio_mask"].to(cfg.device)
            labels_b = batch["labels"].to(cfg.device)
            logits = model(text_b, audio_b, text_mask, audio_mask)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_model(model, val_loader, cfg)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **{k: v for k, v in val_metrics.items() if not isinstance(v, np.ndarray)},
        }
        history.append(row)
        print(f"{mode:>15} epoch {epoch:02d} loss={row['loss']:.4f} val_f1={row['f1_1']:.4f}")

        if val_metrics["f1_1"] > best_val_f1:
            best_val_f1 = val_metrics["f1_1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_on_indices(model, text_hidden, audio_hidden, labels, indices, cfg: PipelineConfig) -> dict:
    loader = make_loader(text_hidden, audio_hidden, labels, indices, cfg, shuffle=False)
    return evaluate_model(model, loader, cfg)


def metric_row(model_name: str, train_lang: str, test_lang: str, metrics: dict) -> dict:
    return {
        "model": model_name,
        "train_lang": train_lang,
        "test_lang": test_lang,
        "accuracy": metrics["accuracy"],
        "precision_1": metrics["precision_1"],
        "recall_1": metrics["recall_1"],
        "f1_1": metrics["f1_1"],
    }


def video_train_val_indices(segments_df: pd.DataFrame, train_lang: str, seed: int, val_size: float = 0.15):
    lang_df = segments_df[segments_df["lang"] == train_lang]
    titles = lang_df["title"].drop_duplicates().tolist()
    if len(titles) < 2:
        idx = lang_df.index.to_numpy()
        labels = lang_df["label"]
        stratify = labels if labels.nunique() > 1 else None
        train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed, stratify=stratify)
        return train_idx, val_idx

    train_titles, val_titles = train_test_split(titles, test_size=val_size, random_state=seed)
    train_idx = lang_df[lang_df["title"].isin(train_titles)].index.to_numpy()
    val_idx = lang_df[lang_df["title"].isin(val_titles)].index.to_numpy()
    return train_idx, val_idx


def video_train_val_test_indices(
    segments_df: pd.DataFrame,
    train_lang: str,
    seed: int,
    val_size: float = 0.15,
    test_size: float = 0.15,
):
    """Split one language into train/val/test, preferably by video title."""
    lang_df = segments_df[segments_df["lang"] == train_lang]
    titles = lang_df["title"].drop_duplicates().tolist()
    holdout_size = val_size + test_size
    if len(titles) < 3:
        idx = lang_df.index.to_numpy()
        labels = lang_df["label"]
        stratify = labels if labels.nunique() > 1 else None
        train_idx, holdout_idx = train_test_split(
            idx,
            test_size=holdout_size,
            random_state=seed,
            stratify=stratify,
        )
        holdout_labels = segments_df.loc[holdout_idx, "label"]
        stratify_holdout = holdout_labels if holdout_labels.nunique() > 1 else None
        relative_test_size = test_size / holdout_size
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=relative_test_size,
            random_state=seed,
            stratify=stratify_holdout,
        )
        return train_idx, val_idx, test_idx

    train_titles, holdout_titles = train_test_split(titles, test_size=holdout_size, random_state=seed)
    relative_test_size = test_size / holdout_size
    val_titles, test_titles = train_test_split(holdout_titles, test_size=relative_test_size, random_state=seed)
    train_idx = lang_df[lang_df["title"].isin(train_titles)].index.to_numpy()
    val_idx = lang_df[lang_df["title"].isin(val_titles)].index.to_numpy()
    test_idx = lang_df[lang_df["title"].isin(test_titles)].index.to_numpy()
    return train_idx, val_idx, test_idx
