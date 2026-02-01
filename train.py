"""Training script for vision-language model projector.

This script trains only the projector component that maps vision features to
language model embedding space. We freeze both the vision encoder and language
decoder to focus learning on the alignment layer.

The training objective is to minimize the distance between projected vision
features and the actual text embeddings, teaching the projector how to translate
visual information into meaningful language representations.

Usage:
    python train.py \\
        --images-dir /path/to/train2017 \\
        --annotations-file /path/to/captions_train2017.json \\
        --epochs 3 \\
        --batch-size 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision_caption import ModelConfig
from vision_caption.data import COCOCaptionDataset
from vision_caption.models import LanguageDecoder, Projector, VisionEncoder


def train_projector(
    images_dir: str,
    annotations_file: str,
    output_dir: str = "checkpoints",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_samples: int | None = None,
    device: str = "cuda",
):
    """Train the vision-to-language projector with frozen encoder and decoder.
    
    Args:
        images_dir: Path to COCO train2017 images
        annotations_file: Path to captions_train2017.json
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Adam learning rate
        max_samples: Optional limit on training samples (for debugging)
        device: Device to train on (cuda/cpu)
    """
    config = ModelConfig()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Determine device and dtype
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Training on {device} with dtype {dtype}")

    # Load vision encoder (frozen)
    print("Loading vision encoder...")
    encoder = VisionEncoder(config.vision_model, dtype=dtype)
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Load language decoder (frozen)
    print("Loading language decoder...")
    decoder = LanguageDecoder(
        model_name=config.language_model,
        dtype=dtype,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
    )
    decoder.to(device)
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    # Initialize trainable projector
    # This is the only component we're training
    print("Initializing projector...")
    projector = Projector(
        vision_dim=encoder.hidden_size,
        text_dim=decoder.hidden_size,
    )
    projector.to(device)
    projector.train()

    # Load COCO dataset
    print("Loading COCO dataset...")
    dataset = COCOCaptionDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        processor=encoder.processor,
        tokenizer=decoder.tokenizer,
        max_length=config.max_new_tokens,
        limit=max_samples,
    )
    
    # Use multiple workers for data loading to maximize GPU utilization
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Dataset size: {len(dataset):,} samples")
    print(f"Training batches per epoch: {len(dataloader):,}")

    # AdamW optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(projector.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine learning rate schedule for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(dataloader)
    )

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Extract frozen features
            with torch.no_grad():
                # Get vision features from frozen encoder
                vision_features = encoder(pixel_values)
                
                # Get target text embeddings from frozen decoder
                text_embeds = decoder.embed_text(input_ids)

            # Project vision features to text space (this is what we're training)
            projected_embeds = projector(vision_features).to(dtype=dtype)

            # Compute alignment loss
            # We use mean pooling to get fixed-size representations
            projected_mean = projected_embeds.mean(dim=1)  # [batch, hidden_dim]
            text_mean = text_embeds.mean(dim=1)  # [batch, hidden_dim]

            # MSE loss encourages projected features to match text embeddings
            loss = F.mse_loss(projected_mean, text_mean)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })

        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        checkpoint_path = output_path / f"projector_epoch_{epoch + 1}.pt"
        torch.save(projector.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = output_path / "projector_final.pt"
    torch.save(projector.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    return projector


def main():
    parser = argparse.ArgumentParser(
        description="Train vision-caption projector on COCO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir", 
        type=str, 
        required=True, 
        help="Path to COCO train2017 images directory"
    )
    parser.add_argument(
        "--annotations-file",
        type=str,
        required=True,
        help="Path to captions_train2017.json",
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="checkpoints", 
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3, 
        help="Learning rate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max training samples (for debugging)",
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    train_projector(
        images_dir=args.images_dir,
        annotations_file=args.annotations_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
