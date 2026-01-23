"""
Image search functionality using vector embeddings.

This module provides functions to search for images using:
- Image similarity (ResNet50 embeddings)
- Text descriptions (CLIP embeddings)
"""

import logging
import re
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Lazy-loaded models
_resnet_model = None
_clip_model = None
_clip_preprocess = None
_device = None
_qdrant_client: Optional[QdrantClient] = None


def get_device() -> str:
    """Get the compute device (CUDA if available, else CPU)."""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_clip_model():
    """Lazy-load CLIP model and preprocessor."""
    global _clip_model, _clip_preprocess
    
    if _clip_model is None:
        import clip
        device = get_device()
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        logger.info("CLIP model loaded for searching")
    
    return _clip_model, _clip_preprocess


def get_resnet_model():
    """Lazy-load ResNet50 model."""
    global _resnet_model
    if _resnet_model is None:
        from tf_keras.applications import ResNet50
        _resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        logger.info("ResNet50 model loaded for searching")
    return _resnet_model


def get_qdrant_client(url: str = "http://localhost:6333") -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client


def preprocess_query(query: str) -> str:
    """
    Convert conversational queries to CLIP-friendly format.
    
    Examples:
        "give me emojis" -> "emoji"
        "show me red icons" -> "red icon"
        "find smiley faces" -> "smiley face"
    
    Args:
        query: Natural language search query
        
    Returns:
        Processed query optimized for CLIP
    """
    # Convert to lowercase
    query = query.lower().strip()
    
    # Remove common conversational prefixes
    conversational_prefixes = [
        r'^(give me|show me|find me|get me|i want|i need|looking for|search for)\s+',
        r'^(can you )?(show|find|get|give)\s+(me\s+)?(some\s+|a\s+|an\s+)?',
    ]
    
    for pattern in conversational_prefixes:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    
    # Remove trailing punctuation and common words
    query = re.sub(r'[\?\.!]+$', '', query)
    query = re.sub(r'\s+(please|thanks|thank you)$', '', query, flags=re.IGNORECASE)
    
    # Handle plural to singular for better matching
    plural_mappings = {
        r'\bemojis\b': 'emoji',
        r'\bicons\b': 'icon',
        r'\bstickers\b': 'sticker',
        r'\bimages\b': 'image',
        r'\bpictures\b': 'picture',
        r'\bphotos\b': 'photo',
        r'\blogos\b': 'logo',
        r'\bbadges\b': 'badge',
        r'\bbuttons\b': 'button',
        r'\bscreenshots\b': 'screenshot',
    }
    
    for plural, singular in plural_mappings.items():
        query = re.sub(plural, singular, query)
    
    # Remove extra whitespace
    query = ' '.join(query.split())
    
    return query


def generate_resnet_embedding(image_path: str) -> Optional[np.ndarray]:
    """Generate ResNet50 embedding for an image."""
    try:
        from tf_keras.preprocessing import image
        from tf_keras.applications.resnet50 import preprocess_input
        
        model = get_resnet_model()
        
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        
        embedding = model.predict(x, verbose=0)
        return embedding.flatten()
        
    except Exception as e:
        logger.error("Error generating ResNet embedding: %s", e)
        return None


def generate_text_embedding(text_query: str) -> Optional[np.ndarray]:
    """
    Generate CLIP embedding for a text query.
    
    Args:
        text_query: Natural language search query
        
    Returns:
        Text embedding vector or None on error
    """
    try:
        import clip
        
        # Preprocess the query
        processed_query = preprocess_query(text_query)
        
        if processed_query != text_query.lower().strip():
            logger.info("Processed query: '%s' → '%s'", text_query, processed_query)
        
        model, _ = get_clip_model()
        device = get_device()
        
        with torch.no_grad():
            text_tokens = clip.tokenize([processed_query]).to(device)
            text_features = model.encode_text(text_tokens)
            text_embedding = text_features.cpu().numpy().flatten()
        
        return text_embedding
        
    except Exception as e:
        logger.error("Error generating text embedding: %s", e)
        return None


def generate_clip_image_embedding(image_path: str) -> Optional[np.ndarray]:
    """Generate CLIP embedding for an image."""
    try:
        model, preprocess = get_clip_model()
        device = get_device()
        
        img = Image.open(image_path).convert("RGB")
        image_input = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_embedding = image_features.cpu().numpy().flatten()
        
        return image_embedding
        
    except Exception as e:
        logger.error("Error generating CLIP image embedding: %s", e)
        return None


def search_by_image(
    query_image_path: str,
    limit: int = 5,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "image_embeddings"
) -> List[dict]:
    """
    Search for similar images using an image as query.
    
    Uses ResNet50 embeddings stored in Qdrant.
    
    Args:
        query_image_path: Path to the query image
        limit: Maximum number of results to return
        qdrant_url: Qdrant server URL
        collection_name: Collection to search
        
    Returns:
        List of search results with image paths and scores
    """
    logger.info("Searching for images similar to: %s", query_image_path)
    
    query_embedding = generate_resnet_embedding(query_image_path)
    
    if query_embedding is None:
        logger.error("Failed to generate embedding for query image")
        return []
    
    try:
        client = get_qdrant_client(qdrant_url)
        
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=limit
        ).points
        
        output = []
        for result in results:
            output.append({
                "image_path": result.payload.get("image_path"),
                "score": result.score
            })
            logger.info("Found: %s (score: %.4f)", result.payload.get("image_path"), result.score)
        
        logger.info("Found %d similar images", len(output))
        return output
        
    except Exception as e:
        logger.error("Error during image search: %s", e)
        return []


def search_by_text(
    text_query: str,
    limit: int = 5,
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "image_embeddings_clip"
) -> List[dict]:
    """
    Search for images using a text description.
    
    Uses CLIP embeddings for text-to-image search.
    
    Args:
        text_query: Natural language search query
        limit: Maximum number of results to return
        qdrant_url: Qdrant server URL
        collection_name: Collection to search (must have CLIP embeddings)
        
    Returns:
        List of search results with image paths and scores
    """
    logger.info("Searching for images matching: '%s'", text_query)
    
    text_embedding = generate_text_embedding(text_query)
    
    if text_embedding is None:
        logger.error("Failed to generate text embedding")
        return []
    
    try:
        client = get_qdrant_client(qdrant_url)
        
        results = client.query_points(
            collection_name=collection_name,
            query=text_embedding.tolist(),
            limit=limit
        ).points
        
        output = []
        for result in results:
            output.append({
                "image_path": result.payload.get("image_path"),
                "score": result.score
            })
            logger.info("Found: %s (score: %.4f)", result.payload.get("image_path"), result.score)
        
        logger.info("Found %d matching images", len(output))
        return output
        
    except Exception as e:
        logger.error("Error during text search: %s", e)
        logger.info("Make sure you have a '%s' collection with CLIP embeddings", collection_name)
        return []


def interactive_search() -> None:
    """Interactive search interface for testing."""
    print("\n" + "=" * 60)
    print("📸 Image Search System")
    print("=" * 60)
    
    while True:
        print("\nChoose search method:")
        print("1. Search by image path")
        print("2. Search by text description")
        print("3. Show example queries")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Enter the path to query image: ").strip()
            num_results = input("Number of results (default 5): ").strip()
            limit = int(num_results) if num_results else 5
            results = search_by_image(image_path, limit)
            
            print(f"\n✓ Found {len(results)} similar images:")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['image_path']} (score: {r['score']:.4f})")
        
        elif choice == "2":
            print("\n💡 Tip: Ask naturally! I'll convert it to a search query.")
            text_query = input("\nEnter your query: ").strip()
            num_results = input("Number of results (default 5): ").strip()
            limit = int(num_results) if num_results else 5
            results = search_by_text(text_query, limit)
            
            print(f"\n✓ Found {len(results)} matching images:")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['image_path']} (score: {r['score']:.4f})")
        
        elif choice == "3":
            print("\n📝 Example Search Queries:")
            print("  • 'emoji' or 'give me emojis'")
            print("  • 'smiley emoji' or 'show me smiley emojis'")
            print("  • 'red icon' or 'find red icons'")
            print("  • 'screenshot' or 'show screenshots'")
            print("\n💡 Tips:")
            print("  • Use simple nouns and adjectives")
            print("  • Colors work well: 'red', 'blue', 'yellow'")
            print("  • Combine attributes: 'yellow smiley emoji'")
        
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    interactive_search()