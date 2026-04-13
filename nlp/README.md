# PoE-EBM on GLUE Benchmarks

This repository implements **PoE-EBM (Product-of-Experts Energy-Based Model) merging** on GLUE benchmarks using Flan-T5 models with LoRA adaptation.

---

## 📂 Checkpoints

Flan-T5 model checkpoints can be downloaded from:

https://github.com/tanganke/subspace_fusion/releases/tag/flan-t5

Place the downloaded checkpoints in:

```
cache/checkpoints/
```

---

## 📁 Datasets

Download all required GLUE datasets and place them in:

```
cache/datasets/
```

Make sure the dataset structure matches the expected format used by the scripts.

---

## 🚀 Running Experiments

### Flan-T5 Base + LoRA (rank 16)

```bash
python3 scripts/flan_t5_cauchy.py models=flan-t5-base peft=lora-16
```

### Flan-T5 Large + LoRA (rank 16)

```bash
python3 scripts/flan_t5_cauchy.py models=flan-t5-large peft=lora-16
```

---

## ⚙️ Notes

- Ensure checkpoints are correctly placed in `cache/checkpoints/`.
- Ensure datasets are correctly placed in `cache/datasets/`.