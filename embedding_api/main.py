import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import numpy as np
from transformers import Qwen3VLProcessor
import logging
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import requests
from typing import Dict, Any
import time
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models/Qwen3-VL-Embedding-2B/scripts'))
from qwen3_vl_embedding import Qwen3VLForEmbedding

# Load .env file if it exists
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Qwen3 Embedding API",
    description="API for generating embeddings using Qwen3-Embedding-4B model",
    version="0.1.0"
)

# Load environment variables
# Try to get MODEL_PATH from environment
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-VL-Embedding-2B")
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    # Fallback to absolute path if environment variable not set
    MODEL_PATH = os.path.join(os.getcwd(), "models", MODEL_NAME)
    logger.info(f"Using fallback MODEL_PATH: {MODEL_PATH}")
DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
API_KEY = os.getenv("API_KEY")

# Dynamic embedding dimension (will be determined automatically from model output)
EMBEDDING_DIMENSION = None

# Image processing configuration
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "512"))
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", os.getenv("MAX_IMAGE_SIZE", "512")))
MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", os.getenv("MAX_IMAGE_SIZE", "512")))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

# Initialize model
logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}...")
try:
    # Set CUDA visible devices if specified
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        logger.info(f"CUDA_VISIBLE_DEVICES set to: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    
    # Verify CUDA availability if using GPU
    if DEVICE == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        DEVICE = "cpu"
    
    # Load Qwen3-VL specific components
    logger.info(f"Loading processor for {MODEL_NAME}...")
    processor = Qwen3VLProcessor.from_pretrained(MODEL_PATH)
    
    logger.info(f"Loading model for {MODEL_NAME}...")
    model = Qwen3VLForEmbedding.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    model.to(DEVICE)
    model.eval()
    
    logger.info(f"Model {MODEL_NAME} loaded successfully on {DEVICE}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Request/Response models
class ImageInput(BaseModel):
    type: str = "image"
    data: Optional[str] = Field(None, description="Base64 encoded image data (for base64 input)")
    url: Optional[str] = Field(None, description="Image URL (for URL input)")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "type": "image",
                    "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/..."
                },
                {
                    "type": "image",
                    "url": "https://example.com/image.jpg"
                }
            ]
        }

class TextInput(BaseModel):
    type: str = "text"
    data: str

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], ImageInput, TextInput, List[Union[ImageInput, TextInput]]]
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
def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and convert to PIL Image."""
    try:
        # Set timeout for image download
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad status codes
        
        # Convert to PIL Image
        image = Image.open(BytesIO(response.content))
        
        # Resize image if needed
        return resize_image(image)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to process downloaded image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data from URL")


def resize_image(image: Image.Image) -> Image.Image:
    """Resize image to optimize performance while maintaining aspect ratio."""
    try:
        # Get original dimensions
        width, height = image.size
        
        # Calculate new dimensions while maintaining aspect ratio
        if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
            # Calculate scaling factor
            scale_x = MAX_IMAGE_WIDTH / width
            scale_y = MAX_IMAGE_HEIGHT / height
            scale = min(scale_x, scale_y)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            logger.info(f"Resizing image from ({width}, {height}) to ({new_width}, {new_height})")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        logger.error(f"Failed to resize image: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to process image")


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 image data to PIL Image."""
    try:
        # Remove data URI prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_str)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        
        # Resize image if needed
        return resize_image(image)
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image data")

def process_input(input_data):
    """Process input data and convert to model-compatible format."""
    if isinstance(input_data, str):
        return [input_data]
    elif isinstance(input_data, list):
        processed = []
        for item in input_data:
            if isinstance(item, str):
                processed.append(item)
            elif isinstance(item, dict) or hasattr(item, 'type'):
                # Get type safely using getattr for objects and direct access for dicts
                item_type = item['type'] if isinstance(item, dict) else getattr(item, 'type', '')
                if item_type == 'text':
                    # Get data safely
                    text_data = item['data'] if isinstance(item, dict) else getattr(item, 'data', '')
                    processed.append(text_data)
                elif item_type == 'image':
                    # Handle both base64 and URL image input
                    # Get image data and url safely
                    image_data = item.get('data') if isinstance(item, dict) else getattr(item, 'data', None)
                    image_url = item.get('url') if isinstance(item, dict) else getattr(item, 'url', None)
                    
                    # Check if data field contains a URL
                    if image_data:
                        # Strip whitespace and check if it's a URL
                        image_data_stripped = image_data.strip()
                        if image_data_stripped.startswith(('http://', 'https://')):
                            # Treat as URL if it starts with http:// or https://
                            image = download_image_from_url(image_data_stripped)
                        else:
                            # Otherwise treat as base64 data
                            image = decode_base64_image(image_data_stripped)
                    elif image_url:
                        image = download_image_from_url(image_url.strip())
                    else:
                        logger.error("Image input must have either 'data' (base64/URL) or 'url' field")
                        raise HTTPException(status_code=400, detail="Image input must have either 'data' (base64/URL) or 'url' field")
                    
                    processed.append(image)
        return processed
    elif isinstance(input_data, dict) or hasattr(input_data, 'type'):
        # Get type safely
        input_type = input_data['type'] if isinstance(input_data, dict) else getattr(input_data, 'type', '')
        if input_type == 'text':
            # Get data safely
            text_data = input_data['data'] if isinstance(input_data, dict) else getattr(input_data, 'data', '')
            return [text_data]
        elif input_type == 'image':
            # Handle both base64 and URL image input
            # Get image data and url safely
            image_data = input_data.get('data') if isinstance(input_data, dict) else getattr(input_data, 'data', None)
            image_url = input_data.get('url') if isinstance(input_data, dict) else getattr(input_data, 'url', None)
            
            # Check if data field contains a URL
            if image_data:
                # Strip whitespace and check if it's a URL
                image_data_stripped = image_data.strip()
                if image_data_stripped.startswith(('http://', 'https://')):
                    # Treat as URL if it starts with http:// or https://
                    image = download_image_from_url(image_data_stripped)
                else:
                    # Otherwise treat as base64 data
                    image = decode_base64_image(image_data_stripped)
            elif image_url:
                image = download_image_from_url(image_url.strip())
            else:
                logger.error("Image input must have either 'data' (base64/URL) or 'url' field")
                raise HTTPException(status_code=400, detail="Image input must have either 'data' (base64/URL) or 'url' field")
            
            return [image]
    return []

def expand_features(embedding: np.ndarray, target_length: int) -> np.ndarray:
    """Expand embedding to target length using simple padding for better performance."""
    if len(embedding) >= target_length:
        return embedding[:target_length]
    
    # Use simple zero padding instead of complex transformers for better performance
    expanded_embedding = np.zeros(target_length)
    expanded_embedding[:len(embedding)] = embedding
    
    # L2 normalization
    norm = np.linalg.norm(expanded_embedding)
    if norm > 0:
        expanded_embedding = expanded_embedding / norm
    
    return expanded_embedding

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint with detailed information."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_image_width": MAX_IMAGE_WIDTH,
        "max_image_height": MAX_IMAGE_HEIGHT
    }

def generate_embedding(input_item, processor, model, device):
    """Generate embedding for a single input item (text or image) using official pooling method."""
    try:
        if isinstance(input_item, str):
            # Text input
            messages = [
                {"role": "user", "content": input_item}
            ]
        else:
            # Image input
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": input_item},
                    {"type": "text", "text": ""}
                ]}
            ]
        
        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(
            text=text,
            images=[input_item] if not isinstance(input_item, str) else None,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate embedding using official pooling method
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use official pooling method: extract the last valid token
            # This is the key difference from average pooling
            if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
                # Find the last non-padding token for each sequence
                attention_mask = inputs['attention_mask']
                # Flip the mask and find the first 1 (which corresponds to the last 1 in original)
                flipped_mask = attention_mask.flip(dims=[1])
                last_token_indices = flipped_mask.argmax(dim=1)
                # Calculate the actual column indices
                col_indices = attention_mask.shape[1] - last_token_indices - 1
                # Create row indices
                row_indices = torch.arange(outputs.last_hidden_state.shape[0], device=device)
                # Extract embeddings for the last valid tokens
                embedding = outputs.last_hidden_state[row_indices, col_indices].squeeze().cpu().numpy()
            else:
                # Fallback: use the last token if no attention mask
                embedding = outputs.last_hidden_state[:, -1, :].squeeze().cpu().numpy()
            
            # Normalize embedding with overflow protection
            try:
                norm = np.linalg.norm(embedding)
                logger.debug(f"Embedding norm: {norm}")
                if norm > 0 and not np.isinf(norm) and not np.isnan(norm):
                    embedding = embedding / norm
                    logger.debug(f"Normalized embedding sample: {embedding[:10]}")
                else:
                    logger.warning(f"Invalid embedding norm: {norm}, using unnormalized embedding")
            except Exception as e:
                logger.error(f"Error normalizing embedding: {e}, using unnormalized embedding")
        
        # Log actual embedding dimension
        logger.info(f"Actual generated embedding dimension: {len(embedding)}")
        
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    _: bool = Depends(verify_token)
):
    """Generate embeddings for the input text(s) or image(s)."""
    start_time = time.time()
    input_processing_time = 0
    embedding_generation_time = 0
    expansion_time = 0
    
    try:
        # Process input
        logger.info(f"Processing input...")
        input_start = time.time()
        processed_input = process_input(request.input)
        input_processing_time = time.time() - input_start
        logger.info(f"Input processed in {input_processing_time:.4f} seconds")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(processed_input)} inputs...")
        embed_start = time.time()
        
        embeddings = []
        for item in processed_input:
            embedding = generate_embedding(item, processor, model, DEVICE)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        embedding_generation_time = time.time() - embed_start
        logger.info(f"Embeddings generated in {embedding_generation_time:.4f} seconds")
        
        # Dynamic embedding dimension - update global EMBEDDING_DIMENSION from actual output
        global EMBEDDING_DIMENSION
        actual_dimension = embeddings.shape[1]
        if EMBEDDING_DIMENSION is None:
            EMBEDDING_DIMENSION = actual_dimension
            logger.info(f"Set dynamic EMBEDDING_DIMENSION to: {EMBEDDING_DIMENSION}")
        elif EMBEDDING_DIMENSION != actual_dimension:
            logger.warning(f"Configured EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION}) doesn't match actual model output ({actual_dimension}). Using actual dimension.")
            EMBEDDING_DIMENSION = actual_dimension
        
        # Expand to target dimension if needed (only if explicitly configured)
        configured_dimension = int(os.getenv("EMBEDDING_DIMENSION", "-1"))
        if configured_dimension > 0 and configured_dimension != actual_dimension:
            logger.info(f"Expanding embeddings from {actual_dimension} to {configured_dimension} dimensions (explicitly configured)")
            expand_start = time.time()
            
            expanded_embeddings = []
            for emb in embeddings:
                expanded = expand_features(emb, configured_dimension)
                expanded_embeddings.append(expanded)
            embeddings = np.array(expanded_embeddings)
            
            expansion_time = time.time() - expand_start
            logger.info(f"Embeddings expanded in {expansion_time:.4f} seconds")
            EMBEDDING_DIMENSION = configured_dimension
        
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
        
        # For multimodal input, we can only count tokens for text inputs
        total_tokens = 0
        for item in processed_input:
            if isinstance(item, str):
                total_tokens += len(enc.encode(item))
            else:  # For images, we don't have a reliable way to count tokens
                total_tokens += 100  # Approximate token count for images
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.4f} seconds")
        
        return {
            "data": response_data,
            "model": MODEL_NAME,
            "object": "list",
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Error generating embeddings after {total_time:.4f} seconds: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with uvicorn programmatically for development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "6008"))
    uvicorn.run("embedding_api.main:app", host="0.0.0.0", port=port, reload=True)
