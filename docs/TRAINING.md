# Training Aura

## 1. Emotion CNN (do this first)

Trains ResNet-50 on AffectNet → 8-way labels + 2048-d embeddings used by the live
continuous art path.

```bat
scripts\train_emotion.bat
```

Or:

```bat
set STORAGE_PATH=.\output
python -m aura.cnn.train --epochs 5 --batch-size 32
```

**Needs**
- CUDA PyTorch on Gabes-PC
- Kaggle credentials for `kagglehub` dataset `noamsegal/affectnet-training-data`

**Writes**
- `output/aura_storage/aura_emotion_classifier.pth`
- `output/aura_storage/confusion_matrix.png`
- `output/aura_storage/emotion_train_metrics.json`

Live signaling loads that checkpoint automatically. If missing, it falls back to
the demo face-color heuristic (`[demo]` in the UI).

## 2. Emotion ↔ art alignment (next)

```bash
python -m aura.clip.train
```

Projects emotion embeddings and art frames into a shared space (InfoNCE).

## 3. Conditioned generator (after 1–2)

Fine-tune / adapt SD-Turbo (LoRA or IP-Adapter) so
`step(current_frame, embedding)` uses the embedding directly instead of
label→prompt mapping.
