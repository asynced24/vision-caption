# Vision Caption (2026)

Inference-first image captioning with a clean, readable architecture.

**Stack:** SigLIP vision encoder + MLP projector + Phi-3 language decoder (LoRA only).

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

The captioner is intentionally simple:
- Pretrained vision encoder + language decoder, no training loop.
- Optional LoRA adapters on the language model.
- A small MLP maps vision features into the decoder’s embedding space.
- Generation happens in one `generate()` method for clarity.

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

- This repo is intentionally inference-first. No training loop is included.
- If you have LoRA adapter weights, set `ModelConfig.lora_adapter_path`.
- First run will download model weights from Hugging Face.
- Sample images live in `sample_images/`.
- `transformers` is pinned to `<5` for Phi-3 compatibility.
