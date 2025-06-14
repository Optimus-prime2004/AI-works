from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
import torch
from config import logger, EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Loads the SentenceTransformer model from cache or disk.
    This function is cached so the model is only loaded once per application run,
    improving performance significantly.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
    # Automatically select CUDA GPU if available, otherwise fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    logger.info(f"Embedding model loaded successfully on device: {device}")
    return model

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculates the cosine similarity between two texts using the loaded embedding model.
    The raw similarity score (0.0 to 1.0) is scaled to a 0-100 score.
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    model = get_embedding_model()
    try:
        # Generate embeddings for both texts
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_scores = util.cos_sim(embedding1, embedding2)
        score = cosine_scores.item() * 100
        
        # Clamp the score to ensure it's within the 0-100 range
        return max(0.0, min(100.0, score))
    except Exception as e:
        logger.error(f"Failed to calculate text similarity: {e}", exc_info=True)
        return 0.0