import os
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Qwen3 Embedding API",
    description="API for generating embeddings using Qwen3-Embedding-4B model",
    version="0.1.0"
)

# Load environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/Qwen3-Embedding-4B")
DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
API_KEY = os.getenv("API_KEY")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "2560"))

# Initialize model
logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}...")
try:
    model = SentenceTransformer(MODEL_PATH, device=DEVICE)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Request/Response models
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "Qwen3-Embedding-4B"

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int
    object: str = "embedding"

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    object: str = "list"
    usage: Usage

# Authentication
async def verify_token(authorization: str = Header(...)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Helper functions
def expand_features(embedding: np.ndarray, target_length: int) -> np.ndarray:
    """Expand embedding to target length using polynomial features and random projection."""
    if len(embedding) >= target_length:
        return embedding[:target_length]
    
    remaining_dims = target_length - len(embedding)
    
    if len(embedding) > 100:
        from sklearn.random_projection import GaussianRandomProjection
        transformer = GaussianRandomProjection(n_components=remaining_dims)
        expanded_part = transformer.fit_transform(embedding.reshape(1, -1)).flatten()
    else:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2)
        expanded_part = poly.fit_transform(embedding.reshape(1, -1)).flatten()
        if len(expanded_part) > remaining_dims:
            expanded_part = expanded_part[:remaining_dims]
        elif len(expanded_part) < remaining_dims:
            expanded_part = np.pad(expanded_part, (0, remaining_dims - len(expanded_part)))
    
    expanded_embedding = np.concatenate([embedding, expanded_part])
    expanded_embedding = expanded_embedding[:target_length]
    
    # L2 normalization
    norm = np.linalg.norm(expanded_embedding)
    if norm > 0:
        expanded_embedding = expanded_embedding / norm
    
    return expanded_embedding

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": "Qwen3-Embedding-4B", "device": DEVICE}

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    _: bool = Depends(verify_token)
):
    """Generate embeddings for the input text(s)."""
    try:
        # Encode input text(s)
        logger.info(f"Generating embeddings for {len(request.input) if isinstance(request.input, list) else 1} text(s)")
        
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        # Generate embeddings
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Expand to target dimension if needed
        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            logger.info(f"Expanding embeddings from {embeddings.shape[1]} to {EMBEDDING_DIMENSION} dimensions")
            expanded_embeddings = []
            for emb in embeddings:
                expanded = expand_features(emb, EMBEDDING_DIMENSION)
                expanded_embeddings.append(expanded)
            embeddings = np.array(expanded_embeddings)
        
        # Prepare response
        response_data = [
            {
                "embedding": emb.tolist(),
                "index": i,
                "object": "embedding"
            }
            for i, emb in enumerate(embeddings)
        ]
        
        # Calculate token usage (approximate)
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(enc.encode(text)) for text in texts)
        
        return {
            "data": response_data,
            "model": request.model,
            "object": "list",
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with uvicorn programmatically for development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "6008"))
    uvicorn.run("embedding_api.main:app", host="0.0.0.0", port=port, reload=True)
