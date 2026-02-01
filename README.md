# Vision Caption

Inference-first image captioning with a clean, readable architecture.

**Stack:** SigLIP vision encoder → MLP projector → Phi-3 decoder + LoRA

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
Phi-3 Decoder + LoRA
  │
  ▼
Caption
```

Built for clarity:
- Frozen pretrained encoder + decoder — no training loop in this repo
- LoRA adapters on the language model for efficient fine-tuning
- MLP bridges vision features into the decoder's embedding space
- Single `generate()` method makes the inference path obvious

This is a compact VLM that's easy to reason about, extend, and demo.

---

## Quick Start

```bash
pip install -e .
python app.py
```

Gradio opens a local page (usually `http://127.0.0.1:7860`) where you can upload an image.

---

## Minimal API

```python
from vision_caption import load_model

model = load_model()
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
  inference.py
  config.py

app.py
notebooks/demo.ipynb
```

---

## Notes

- Inference-first — no training loop included
- LoRA adapters load from `ModelConfig.lora_adapter_path` if provided
- First run downloads ~8GB of model weights from Hugging Face
- Runs best on GPU; CPU needs 16GB+ RAM
