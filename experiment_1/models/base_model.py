from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def __init__(self, model_config):
        """Initialize the model with its configuration."""
        pass
    
    @abstractmethod
    def generate(self, prompt, image_path):
        """Generate a response for the given prompt and image."""
        pass
    
    @abstractmethod
    def process_output(self, output):
        """Process the model output to extract the score."""
        pass 