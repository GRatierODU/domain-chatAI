import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
from typing import Dict, List, Optional, Tuple
import io
import base64
import logging

logger = logging.getLogger(__name__)

class VisualAnalyzer:
    """
    Analyzes visual content from web pages
    """
    
    def __init__(self):
        self._load_models()
        
    def _load_models(self):
        """Load visual analysis models"""
        try:
            # BLIP for image captioning and understanding
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.blip_model = self.blip_model.cuda()
                
            logger.info("Visual analysis models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load visual models: {e}")
            # Fallback to CPU mode
            self.blip_model = None
            
    async def analyze_image(self, image_bytes: bytes, context: Dict = {}) -> Dict:
        """
        Analyze an image and extract information
        """
        if not self.blip_model:
            return {"error": "Visual model not loaded"}
            
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Generate caption
            inputs = self.blip_processor(image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            # Generate description
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Analyze image properties
            analysis = {
                "caption": caption,
                "size": image.size,
                "mode": image.mode,
                "has_text": self._detect_text_regions(image),
                "dominant_colors": self._extract_dominant_colors(image),
                "image_type": self._classify_image_type(caption, context)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}
            
    def _detect_text_regions(self, image: Image) -> bool:
        """
        Simple text detection (checks for high contrast regions)
        """
        # Convert to grayscale
        gray = image.convert('L')
        # Get histogram
        hist = gray.histogram()
        # Check for bimodal distribution (text usually creates high contrast)
        return max(hist) > sum(hist) / len(hist) * 5
        
    def _extract_dominant_colors(self, image: Image, num_colors: int = 3) -> List[str]:
        """
        Extract dominant colors from image
        """
        # Resize for faster processing
        image = image.resize((150, 150))
        # Convert to RGB
        image = image.convert('RGB')
        # Get colors
        pixels = image.getdata()
        # Simple color quantization
        color_counts = {}
        for pixel in pixels:
            # Round to nearest 32 to reduce color space
            color = tuple(c // 32 * 32 for c in pixel)
            color_counts[color] = color_counts.get(color, 0) + 1
            
        # Get top colors
        top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:num_colors]
        return [f"rgb{color}" for color, _ in top_colors]
        
    def _classify_image_type(self, caption: str, context: Dict) -> str:
        """
        Classify the type of image based on caption and context
        """
        caption_lower = caption.lower()
        
        if any(word in caption_lower for word in ['product', 'item', 'sale']):
            return 'product'
        elif any(word in caption_lower for word in ['logo', 'brand']):
            return 'logo'
        elif any(word in caption_lower for word in ['banner', 'header', 'hero']):
            return 'banner'
        elif any(word in caption_lower for word in ['icon', 'button']):
            return 'ui_element'
        elif any(word in caption_lower for word in ['chart', 'graph', 'diagram']):
            return 'data_visualization'
        elif any(word in caption_lower for word in ['person', 'people', 'team']):
            return 'people'
        else:
            return 'general'