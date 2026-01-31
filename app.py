from pathlib import Path

import gradio as gr
from vision_caption import ModelConfig, load_model


def _load_examples(default_prompt: str):
    sample_dir = Path(__file__).parent / "sample_images"
    if not sample_dir.exists():
        return None
    image_paths = sorted(sample_dir.glob("*.png"))
    return [[str(path), default_prompt] for path in image_paths]


def build_demo():
    config = ModelConfig()
    model = load_model(config)
    examples = _load_examples(config.prompt)

    def caption_image(image, prompt):
        if image is None:
            return "Please upload an image."
        return model.generate(image=image, prompt=prompt)

    return gr.Interface(
        fn=caption_image,
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Textbox(value=config.prompt, label="Prompt"),
        ],
        outputs=gr.Textbox(label="Caption"),
        title="Vision Caption",
        description="SigLIP + Phi-3 + LoRA. Inference-first captioning demo.",
        examples=examples,
        allow_flagging="never",
    )


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
