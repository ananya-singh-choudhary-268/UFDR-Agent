"""
Image embedding generation using ResNet50 and FaceNet.

This module provides functions to generate embeddings for images
using pre-trained deep learning models for forensic analysis.
"""

import logging
import os
import uuid
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)

# Lazy-loaded models to avoid slow imports
_resnet_model = None
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client(url: str = "http://localhost:6333") -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client


def get_resnet_model():
    """Lazy-load ResNet50 model."""
    global _resnet_model
    if _resnet_model is None:
        from tf_keras.applications import ResNet50
        _resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        logger.info("ResNet50 model loaded")
    return _resnet_model


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str = "image_embeddings",
    vector_size: int = 2048
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


def generate_resnet_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Generate image embedding using ResNet-50 for general images.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Flattened embedding vector (2048 dimensions) or None on error
    """
    try:
        from tf_keras.preprocessing import image
        from tf_keras.applications.resnet50 import preprocess_input
        
        model = get_resnet_model()
        
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        
        # Generate embedding
        embedding = model.predict(x, verbose=0)
        return embedding.flatten()
        
    except Exception as e:
        logger.error("Error processing image %s: %s", image_path, e)
        return None


def generate_facenet_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Generate image embedding using FaceNet for face images.
    
    Args:
        image_path: Path to the face image file
        
    Returns:
        Embedding vector (128 dimensions) or None on error
    """
    try:
        from deepface import DeepFace
        
        embedding = DeepFace.represent(image_path, model_name='Facenet')[0]['embedding']
        return np.array(embedding)
        
    except Exception as e:
        logger.error("Error processing face image %s: %s", image_path, e)
        return None


def insert_embedding_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    image_path: str,
    embedding: np.ndarray
) -> str:
    """
    Insert embedding into Qdrant vector database.
    
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
    
    logger.debug("Inserted embedding for image: %s", image_path)
    return image_id


def process_images_from_folder(
    folder_path: str,
    use_facenet: bool = False,
    max_images: Optional[int] = None,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "image_embeddings"
) -> List[str]:
    """
    Process images from a folder and insert embeddings into Qdrant.
    
    Args:
        folder_path: Path to folder containing images
        use_facenet: Use FaceNet (for faces) instead of ResNet50 (general)
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
    vector_size = 128 if use_facenet else 2048
    ensure_collection_exists(client, collection_name, vector_size)
    
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    processed = []
    
    for filename in os.listdir(folder_path):
        if max_images and len(processed) >= max_images:
            logger.info("Reached limit of %d images", max_images)
            break
            
        if not filename.lower().endswith(image_extensions):
            continue
            
        image_path = os.path.join(folder_path, filename)
        
        # Generate embedding
        if use_facenet:
            embedding = generate_facenet_embedding(image_path)
        else:
            embedding = generate_resnet_embedding(image_path)
        
        if embedding is not None:
            insert_embedding_to_qdrant(client, collection_name, image_path, embedding)
            processed.append(image_path)
            logger.info("Processed: %s", filename)
        else:
            logger.warning("Skipped: %s", filename)
    
    logger.info("Successfully processed %d images", len(processed))
    return processed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate image embeddings")
    parser.add_argument("folder", help="Path to image folder")
    parser.add_argument("--facenet", action="store_true", help="Use FaceNet for faces")
    parser.add_argument("--max", type=int, default=10, help="Max images to process")
    parser.add_argument("--qdrant", default="http://localhost:6333", help="Qdrant URL")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    processed = process_images_from_folder(
        folder_path=args.folder,
        use_facenet=args.facenet,
        max_images=args.max,
        qdrant_url=args.qdrant
    )
    
    print(f"\n✓ Successfully processed {len(processed)} images!")
