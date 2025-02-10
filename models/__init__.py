from .base_model import BaseModel
# from .mplug_owl2 import MPLUGOwl2Model
from .llava_1_5 import LLAVA1_5
from .llava_1_6 import LLAVA1_6
from .idefics_9b_instruct import IDEFICS9bModel
from .internlm_xc2_vl import InternLMXC2Model

# Map model names to their implementations
MODEL_REGISTRY = {
    "llava-v1.5-7b": LLAVA1_5,
    "llava-v1.6-vicuna-7b": LLAVA1_6,
    "idefics-9b-instruct": IDEFICS9bModel,
    "internlm-xcomposer2-7b": InternLMXC2Model,
    # "mplug-owl2-llama2-7b": MPLUGOwl2Model,
}

def load_model(model_config):
    """Factory function to load the appropriate model."""
    model_name = model_config["model_name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_REGISTRY[model_name](model_config) 