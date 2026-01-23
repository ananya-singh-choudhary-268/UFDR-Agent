"""
CLIP-based image embedding generation.

This module provides functions to generate embeddings for images
using OpenAI's CLIP model for text-to-image search capabilities.
"""

import logging
import os
import uuid
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)

# Lazy-loaded models
_clip_model = None
_clip_preprocess = None
_device = None
_qdrant_client: Optional[QdrantClient] = None

# CLIP constants
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_EMBEDDING_DIM = 512


def get_device() -> str:
    """Get the compute device (CUDA if available, else CPU)."""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", _device)
    return _device


def get_clip_model():
    """Lazy-load CLIP model and preprocessor."""
    global _clip_model, _clip_preprocess
    
    if _clip_model is None:
        import clip
        device = get_device()
        _clip_model, _clip_preprocess = clip.load(CLIP_MODEL_NAME, device=device)
        logger.info("CLIP model loaded: %s", CLIP_MODEL_NAME)
    
    return _clip_model, _clip_preprocess


def get_qdrant_client(url: str = "http://localhost:6333") -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str = "image_embeddings_clip",
    vector_size: int = CLIP_EMBEDDING_DIM
) -> None:
    """
    Ensure the Qdrant collection exists, creating it if necessary.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        vector_size: Dimension of embedding vectors
    """
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info("Created collection: %s", collection_name)


def generate_clip_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Generate CLIP embedding for an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Normalized embedding vector (512 dimensions) or None on error
    """
    try:
        model, preprocess = get_clip_model()
        device = get_device()
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        image_input = preprocess(img).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten()
        
        return embedding
        
    except Exception as e:
        logger.error("Error processing image %s: %s", image_path, e)
        return None


def insert_embedding_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    image_path: str,
    embedding: np.ndarray
) -> str:
    """
    Insert CLIP embedding into Qdrant vector database.
    
    Args:
        client: Qdrant client instance
        collection_name: Target collection name
        image_path: Path to the original image (stored as metadata)
        embedding: Embedding vector to store
        
    Returns:
        Generated UUID for the inserted point
    """
    image_id = str(uuid.uuid4())
    
    client.upsert(
        collection_name=collection_name,
        points=[{
            "id": image_id,
            "vector": embedding.tolist(),
            "payload": {"image_path": image_path}
        }]
    )
    
    logger.debug("Inserted CLIP embedding for: %s", os.path.basename(image_path))
    return image_id


def process_images_with_clip(
    folder_path: str,
    max_images: Optional[int] = None,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "image_embeddings_clip"
) -> List[str]:
    """
    Process images from a folder and insert CLIP embeddings into Qdrant.
    
    Args:
        folder_path: Path to folder containing images
        max_images: Maximum number of images to process (None for all)
        qdrant_url: Qdrant server URL
        collection_name: Target collection name
        
    Returns:
        List of successfully processed image paths
    """
    if not os.path.isdir(folder_path):
        logger.error("Invalid folder path: %s", folder_path)
        return []
    
    client = get_qdrant_client(qdrant_url)
    ensure_collection_exists(client, collection_name)
    
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    processed = []
    skipped = 0
    
    logger.info("Starting CLIP embedding generation")
    logger.info("Folder: %s", folder_path)
    logger.info("Max images: %s", max_images or "unlimited")
    
    for filename in os.listdir(folder_path):
        if max_images and len(processed) >= max_images:
            logger.info("Reached limit of %d images", max_images)
            break
            
        if not filename.lower().endswith(image_extensions):
            skipped += 1
            continue
            
        image_path = os.path.join(folder_path, filename)
        
        embedding = generate_clip_embedding(image_path)
        
        if embedding is not None:
            insert_embedding_to_qdrant(client, collection_name, image_path, embedding)
            processed.append(image_path)
            logger.info("[%d] Processed: %s", len(processed), filename[:50])
        else:
            skipped += 1
            logger.warning("Skipped: %s", filename)
    
    logger.info("Processing complete: %d processed, %d skipped", len(processed), skipped)
    return processed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CLIP image embeddings")
    parser.add_argument("folder", help="Path to image folder")
    parser.add_argument("--max", type=int, default=10, help="Max images to process")
    parser.add_argument("--qdrant", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="image_embeddings_clip", help="Collection name")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    processed = process_images_with_clip(
        folder_path=args.folder,
        max_images=args.max,
        qdrant_url=args.qdrant,
        collection_name=args.collection
    )
    
    print(f"\n✅ Successfully processed {len(processed)} images!")
    print("You can now use text search with searching.py!")