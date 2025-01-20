from .base_model import BaseModel
from .mplug_owl import MPLUGOwlModel

# Map model names to their implementations
MODEL_REGISTRY = {
    "mPLUG-Owl3-7B-240728": MPLUGOwlModel,
}

def load_model(model_config):
    """Factory function to load the appropriate model."""
    model_name = model_config["model_name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_REGISTRY[model_name](model_config) 