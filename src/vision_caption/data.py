"""COCO dataset loader for vision-language model training.

This module handles loading and preprocessing of COCO Captions dataset,
which is widely used for image captioning tasks. Each image has 5 human-annotated
captions describing its content.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class COCOCaptionDataset(Dataset):
    """COCO Captions dataset for training vision-language projectors.
    
    The dataset pairs images with their human-written captions. During training,
    we use these pairs to teach the projector how to map vision features into
    the language model's embedding space.
    
    Args:
        images_dir: Directory containing COCO image files
        annotations_file: Path to COCO captions_train2017.json or similar
        processor: Vision processor for image preprocessing (from vision encoder)
        tokenizer: Text tokenizer (from language decoder)
        max_length: Maximum caption length in tokens
        limit: Optional limit on dataset size (useful for quick experiments)
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        processor,
        tokenizer,
        max_length: int = 64,
        limit: Optional[int] = None,
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load COCO annotations file
        with open(annotations_file) as f:
            coco_data = json.load(f)

        # Build mapping from image_id to captions
        # Each image typically has 5 captions from different annotators
        self.image_to_captions = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            if image_id not in self.image_to_captions:
                self.image_to_captions[image_id] = []
            self.image_to_captions[image_id].append(caption)

        # Build mapping from image_id to filename
        self.image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}

        # Create training samples: (image_id, caption) pairs
        # We use all available captions to maximize training data
        self.samples = []
        for image_id, captions in self.image_to_captions.items():
            if image_id in self.image_id_to_file:
                # Use all captions for each image (data augmentation)
                for caption in captions:
                    self.samples.append((image_id, caption))

        # Optionally limit dataset size for debugging or quick iterations
        if limit:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Load and preprocess a single training sample."""
        image_id, caption = self.samples[idx]
        image_path = self.images_dir / self.image_id_to_file[image_id]

        # Load image and convert to RGB (handles grayscale/RGBA)
        image = Image.open(image_path).convert("RGB")
        
        # Process image using vision encoder's preprocessor
        # This handles resizing, normalization, and tensor conversion
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        # Tokenize caption with padding and truncation
        tokens = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "caption": caption,  # Keep raw text for logging/debugging
        }
