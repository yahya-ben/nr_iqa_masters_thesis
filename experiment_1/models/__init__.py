from .base_model import BaseModel
from .mplug_owl import MPLUGOwlModel
from .llava_1_5 import LLAVA1_5
from .llava_1_6 import LLAVA1_6

# Map model names to their implementations
MODEL_REGISTRY = {
    "mPLUG-Owl3-7B-240728": MPLUGOwlModel,
    "llava-v1.5-7b": LLAVA1_5,
    "llava-v1.6-vicuna-7b": LLAVA1_6,
}

def load_model(model_config):
    """Factory function to load the appropriate model."""
    model_name = model_config["model_name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_REGISTRY[model_name](model_config) 