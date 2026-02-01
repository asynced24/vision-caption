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

**Google Colab:**
```python
!git clone https://github.com/asynced24/vision-caption.git
%cd vision-caption
!pip install -e .
!python colab_app.py
```

Opens a public Gradio link you can share.

---

## Training

Train the projector on COCO Captions (full dataset: 118k images):

```python
# In Google Colab (recommended)
# Open notebooks/train_colab.ipynb and run all cells
```

**What gets trained:** Only the MLP projector (~10M params)  
**What stays frozen:** Vision encoder + language decoder  
**Dataset:** COCO train2017 (download handled automatically)  
**Time:** ~2-3 hours on free T4 GPU  

After training, download `projector_final.pt` and set:
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
