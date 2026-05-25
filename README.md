# Cross-Lingual Multimodal Humor Prediction

This repository contains a sample pipeline for testing whether adding audio improves cross-lingual transfer in laughter-associated humor prediction.

The current experiment trains on English and evaluates on held-out English, French, Spanish, and Hungarian:

- `text_only`: Qwen3 text hidden states only
- `text_audio_concat`: Qwen3 text hidden states + Whisper audio hidden states with concat fusion
- `text_audio_cross_attention`: Qwen3 text hidden states + Whisper audio hidden states with cross-attention fusion

Laughter annotations are used as labels only. They are not used as an input modality or as a laughter-only baseline.

## Repository Structure

```text
asr-output/
  en/
  fr/
  es/
  hu/
```

ASR transcript JSON files. Each file contains transcript chunks with text and timestamps.

```text
cleaned-data/
  en/
  fr/
  es/
  hu/
```

Cleaned audio files (not committed) should be saved in this folder. Audio files are expected to preserve the original title and include only the foreground audio.

```text
laughter_crowd/
  en/
  fr/
  es/
  hu/
```

Precomputed laughter annotation JSON files. Each annotation entry should include `start_sec` and `end_sec`. These intervals are used to create segment-level binary labels.

```text
pipeline/
```

Python modules for data matching, label loading, embedding extraction, fusion models, and training.

Key files:

- `pipeline/config.py`: shared configuration
- `pipeline/data.py`: transcript/audio matching and segment dataframe construction
- `pipeline/laughter.py`: laughter annotation loader
- `pipeline/embeddings.py`: Qwen3 and Whisper feature extraction
- `pipeline/models.py`: text-only, concat fusion, and cross-attention fusion models
- `pipeline/train.py`: training and evaluation helpers
- `pipeline/experiment.py`: cross-lingual dataset construction

```text
pipeline-test.ipynb
```

Main runner notebook for the cross-lingual experiment.

## How To Run

1. Open `pipeline-test.ipynb`.
2. Run the install/import cells.
3. Check the config cell:

```python
TRAIN_LANG = "en"
TEST_LANGS = ["fr", "es", "hu"]
```

4. Run the dataset construction cells.
5. Run feature extraction. The pipeline will cache Qwen3 and Whisper hidden states under `pipeline_artifacts/`.
6. Run the training cell.

The final cross-lingual results are saved to:

```text
pipeline_artifacts/crosslingual_transfer_results.csv
```

## Notes

- `cleaned-data/` currently contains sample data only.
- Large generated artifacts such as `pipeline_artifacts/`, `__pycache__/`, and notebook checkpoint folders should not be committed.
- The current label source is `laughter_crowd/`, not RMS energy.
- The main comparison is within each test language: `text_only` vs `text_audio_concat` and `text_audio_cross_attention`.

