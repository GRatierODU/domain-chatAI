import torch
import psutil
import GPUtil
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages model loading and resource allocation
    """
    
    @staticmethod
    def get_available_memory() -> Dict[str, float]:
        """
        Get available system memory
        """
        # CPU RAM
        cpu_memory = psutil.virtual_memory()
        
        # GPU memory
        gpu_memory = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory = sum(gpu.memoryFree for gpu in gpus)
        except:
            pass
            
        return {
            'cpu_gb': cpu_memory.available / (1024**3),
            'gpu_gb': gpu_memory / 1024 if gpu_memory else 0
        }
        
    @staticmethod
    def can_load_model(model_size_gb: float) -> bool:
        """
        Check if model can be loaded
        """
        memory = ModelManager.get_available_memory()
        
        if torch.cuda.is_available():
            return memory['gpu_gb'] >= model_size_gb * 1.2  # 20% overhead
        else:
            return memory['cpu_gb'] >= model_size_gb * 1.5  # 50% overhead for CPU