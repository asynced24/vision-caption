from pathlib import Path

import gradio as gr
from vision_caption import ModelConfig, load_model


def _load_examples():
    sample_dir = Path(__file__).parent / "sample_images"
    if not sample_dir.exists():
        return None
    image_paths = sorted(sample_dir.glob("*.png"))
    return [str(path) for path in image_paths]


def build_demo():
    config = ModelConfig()
    model = load_model(config)
    examples = _load_examples()

    def caption_image(image, prompt):
        if image is None:
            return "Please upload an image."
        return model.generate(image=image, prompt=prompt)

    with gr.Blocks(title="Vision Caption") as demo:
        gr.Markdown("## Vision Caption")
        gr.Markdown("SigLIP + Phi-3 + LoRA — inference-first captioning")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Image")
            prompt_input = gr.Textbox(value=config.prompt, label="Prompt")

        output = gr.Textbox(label="Caption")
        run_btn = gr.Button("Generate")
        run_btn.click(fn=caption_image, inputs=[image_input, prompt_input], outputs=output)

        if examples:
            gr.Markdown("### Examples")
            gr.Examples(examples=examples, inputs=image_input)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
