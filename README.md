# Hyperspectral Antimicrobial Resistance Classification

Research code for classifying antimicrobial-resistance labels in *Staphylococcus* spp. from hyperspectral image cubes. It provides a spectral 1D CNN and a cube-based 3D CNN, train-only normalization, separate train/validation files, and held-out evaluation.

This repository contains no experimental data and reports no model results. The original measurements and sample-level split must be supplied by the researcher before training. Do not use this research code for clinical decisions.

## Data format

Create `.npz` files with two arrays:

```python
import numpy as np

np.savez_compressed(
    "train.npz",
    cubes=cubes,   # float array shaped (samples, height, width, bands)
    labels=labels, # one string or integer class label per cube
)
```

Each cube must have the same shape. Values must be finite. Keep biological samples, acquisition sessions, and repeated measurements in only one split; split by the independent experimental unit before exporting `train.npz`, `validation.npz`, and `test.npz`. This prevents nearly identical measurements from appearing in training and evaluation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Train

```bash
hsi-amr train \
  --train data/train.npz \
  --validation data/validation.npz \
  --model spectral \
  --checkpoint artifacts/spectral.pt
```

Use `--model cube3d` for the 3D CNN. The spectral model averages each cube spatially and classifies its band profile. The 3D model uses spatial and spectral structure together. Normalization statistics are fitted on training cubes only and stored with the model.

## Evaluate held-out data

```bash
hsi-amr evaluate \
  --checkpoint artifacts/spectral.pt \
  --data data/test.npz \
  --output artifacts/test-metrics.json
```

The report includes balanced accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix. It contains measured results only when real data are supplied.

## Limitations

- No dataset, trained weights, or performance claim is included.
- The model assumes all cubes use the same wavelength bands and acquisition protocol.
- Report the split unit, class counts, acquisition conditions, and any preprocessing in a study. Validate results on independent samples and acquisition sessions.
- A high score on a small or repeated-sample split is not evidence of clinical usefulness.

## Português

Pipeline de pesquisa para classificar rótulos de resistência antimicrobiana em *Staphylococcus* spp. a partir de cubos hiperespectrais. Inclui CNN 1D para perfis espectrais e CNN 3D para cubos, normalização calculada somente no treino e avaliação em conjunto separado.

O repositório não contém dados experimentais nem resultados de treinamento. Para treinar, forneça arquivos `.npz` com `cubes` no formato `(amostras, altura, largura, bandas)` e `labels` com um rótulo por cubo. Separe por amostra biológica ou sessão de aquisição antes de gerar treino, validação e teste. Este código é para pesquisa e não deve ser usado para decisões clínicas.

