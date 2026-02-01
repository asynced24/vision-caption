from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from vision_caption import ModelConfig, load_model
from train import train_projector


def _ensure_path_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


def _load_local_paths() -> tuple[Optional[str], Optional[str]]:
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "local_paths.json"
    images_dir = os.environ.get("COCO_IMAGES_DIR")
    annotations_file = os.environ.get("COCO_ANNOTATIONS_FILE")

    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text())
            images_dir = images_dir or data.get("images_dir")
            annotations_file = annotations_file or data.get("annotations_file")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in {config_path}: {exc}") from exc

    return images_dir, annotations_file


def _run_train(args: argparse.Namespace) -> None:
    images_dir = args.images_dir
    annotations_file = args.annotations_file
    if not images_dir or not annotations_file:
        images_dir, annotations_file = _load_local_paths()

    if not images_dir or not annotations_file:
        raise SystemExit(
            "Missing COCO paths. Pass --images-dir/--annotations-file or create local_paths.json."
        )

    images_dir = Path(images_dir)
    annotations_file = Path(annotations_file)

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
    train_parser.add_argument("--images-dir", default=None, help="Path to COCO train2017 images.")
    train_parser.add_argument(
        "--annotations-file",
        default=None,
        help="Path to captions_train2017.json.",
    )
    train_parser.add_argument("--output-dir", default="checkpoints", help="Where to save weights.")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
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
