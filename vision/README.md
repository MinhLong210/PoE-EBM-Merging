# PoE-MoE Merging

This repository implements **PoE-MoE**, built on top of:

> **Model Merging with SVD to Tie the KnOTS**  
> https://github.com/gstoica27/KnOTS

---

## 📁 Data Preparation

Download all required datasets and place them under:

```
/data
```

Make sure the directory structure matches what is expected by the original KnOTS repository.

---

## 📂 Checkpoints

### Fully-Finetuned (FFT) Checkpoints

Download from Google Drive:

https://drive.google.com/drive/folders/1fzHAN3v0qDJuHiD3EkQ76K0sc1cCO_S-

Save them to:

```
checkpoints_model_merging/FFT_checkpoints/
```

---

### LoRA Checkpoints

LoRA checkpoints are available on HuggingFace:

https://huggingface.co/collections/hoffman-lab/knots-model-merging-with-svd

Download the required LoRA checkpoints and place them according to your experiment configuration.

---

## 🚀 Generate CLIP Heads

Before evaluation, generate CLIP classification heads:

```bash
python3 -m dataset.parsing.generate_clip_heads
```

The generated heads will be saved in:

```
checkpoints_model_merging/heads/
```

---

## 🧪 Evaluation

To evaluate PoE-MoE merging performance:

1. Open:

```
eval_scripts/vision_pertask.py
```

2. Modify the configuration:

```python
CONFIG_NAME = "your_config_name"
```

3. Run evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m eval_scripts.vision_pertask
```

---

## ⚙️ Configuration

All experiment settings (merging strategy, task setup, LoRA configuration, etc.) are controlled via:

```
configs/
```

Modify `CONFIG_NAME` to switch between different merging methods and experimental setups.

---

## 📌 Notes

- Ensure datasets are correctly placed in `/data`.
- Ensure FFT checkpoints and LoRA checkpoints are in their respective directories.
- Generate CLIP heads before running evaluation.
