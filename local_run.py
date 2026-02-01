from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from vision_caption import ModelConfig, load_model
from train import train_projector


def _ensure_path_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


def _run_train(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir)
    annotations_file = Path(args.annotations_file)

    _ensure_path_exists(images_dir, "Images directory")
    _ensure_path_exists(annotations_file, "Annotations file")

    train_projector(
        images_dir=str(images_dir),
        annotations_file=str(annotations_file),
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        device=args.device,
    )


def _resolve_projector_path(explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        return explicit_path
    default_path = Path("checkpoints") / "projector_final.pt"
    if default_path.is_file():
        return str(default_path)
    return None


def _run_caption(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    _ensure_path_exists(image_path, "Image")

    config = ModelConfig()
    projector_path = _resolve_projector_path(args.projector_path)
    if projector_path:
        config.projector_path = projector_path

    model = load_model(config=config, device=args.device)
    caption = model.generate(
        image=str(image_path),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(caption)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local runner for training or inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the projector locally.")
    train_parser.add_argument("--images-dir", required=True, help="Path to COCO train2017 images.")
    train_parser.add_argument(
        "--annotations-file",
        required=True,
        help="Path to captions_train2017.json.",
    )
    train_parser.add_argument("--output-dir", default="checkpoints", help="Where to save weights.")
    train_parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for faster experiments.",
    )
    train_parser.add_argument("--device", default="cuda", help="cuda or cpu.")

    caption_parser = subparsers.add_parser("caption", help="Generate a caption for an image.")
    caption_parser.add_argument("image", help="Path to a local image.")
    caption_parser.add_argument(
        "--projector-path",
        default=None,
        help="Path to trained projector weights (defaults to checkpoints/projector_final.pt).",
    )
    caption_parser.add_argument("--device", default=None, help="cuda or cpu.")
    caption_parser.add_argument("--prompt", default=None, help="Optional prompt override.")
    caption_parser.add_argument("--max-new-tokens", type=int, default=None)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "train":
        _run_train(args)
    else:
        _run_caption(args)


if __name__ == "__main__":
    main()
