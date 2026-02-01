# Vision Caption

Inference-first image captioning with a clean, readable architecture.

**Stack:** SigLIP vision encoder → MLP projector → Qwen2-1.5B decoder + LoRA

---

## Architecture

```
Image
  │
  ▼
SigLIP Vision Encoder
  │   (patch embeddings)
  ▼
MLP Projector (vision → text space)
  │
  ▼
Qwen2-1.5B Decoder + LoRA
  │
  ▼
Caption
```

Built for clarity:
- Frozen pretrained encoder + decoder
- Trainable MLP projector maps vision to language space
- LoRA-ready for efficient language model fine-tuning
- Clean training + inference separation

This is a complete VLM pipeline that's easy to reason about, train, and demo.

---

## Quick Start

**Local:**
```bash
pip install -e .
python app.py
```

Gradio opens at `http://127.0.0.1:7860`.

**Google Colab (demo only):**
```python
!git clone https://github.com/asynced24/vision-caption.git
%cd vision-caption
%pip install -e .
!python colab_app.py
```

Opens a public Gradio link. **Note:** Requires a trained projector (see Training section).

---

## Training

Train the projector on COCO Captions (full dataset: 118k images):

**In Google Colab:**

1. Go to https://colab.research.google.com/
2. File → Open notebook → GitHub → `asynced24/vision-caption` → `notebooks/train_colab.ipynb`
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells (Ctrl+F9)

**What happens:**
- Downloads COCO train2017 (~19GB images + annotations)
- Trains projector for ~2-3 hours
- Saves `checkpoints/projector_final.pt`
- Downloads trained weights to your machine

**After training:**
```python
config = ModelConfig()
config.projector_path = "checkpoints/projector_final.pt"
model = load_model(config)
```

---

## Minimal API

```python
from vision_caption import ModelConfig, load_model

# With trained projector
config = ModelConfig()
config.projector_path = "checkpoints/projector_final.pt"
model = load_model(config)

caption = model.generate("path/to/image.jpg")
print(caption)
```

---

## Project Layout

```
src/vision_caption/
  models/
    vision_encoder.py
    projector.py
    language_decoder.py
    vlm.py
  data.py
  inference.py
  config.py

train.py
app.py
colab_app.py
notebooks/
  train_colab.ipynb
  demo.ipynb
```

---

## Notes

- **Projector must be trained** — run `notebooks/train_colab.ipynb` to train on COCO
- LoRA adapters load from `ModelConfig.lora_adapter_path` if provided
- First run downloads ~4GB of pretrained weights from Hugging Face
- **Training:** Full COCO train2017 (~118k images) takes ~2-3 hours on Colab T4
