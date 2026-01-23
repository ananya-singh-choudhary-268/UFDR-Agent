"""
Image embeddings module for UFDR forensic analysis.

Provides functionality to generate and search image embeddings
using ResNet50, FaceNet, and CLIP models.
"""

from .generate_embeddings import (
    generate_resnet_embedding,
    generate_facenet_embedding,
    process_images_from_folder,
)
from .generating_embeddings_clips import (
    generate_clip_embedding,
    process_images_with_clip,
)
from .searching import (
    search_by_image,
    search_by_text,
)

__all__ = [
    "generate_resnet_embedding",
    "generate_facenet_embedding",
    "generate_clip_embedding",
    "process_images_from_folder",
    "process_images_with_clip",
    "search_by_image",
    "search_by_text",
]
