from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from .base_model import BaseModel

class MPLUGOwlModel(BaseModel):
    def __init__(self, model_config):
        """Initialize mPLUG-Owl model."""
        self.model_path = model_config["model_path"]
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.processor = self.model.init_processor(self.tokenizer)
        
        # Move model to GPU if available
        self.model.to('cuda')
    
    def generate(self, prompt, image_path):
        """Generate response using mPLUG-Owl."""
        # Load and process image
        image = Image.open(image_path)
        
        # Format messages
        messages = [
            {"role": "user", "content": f"<|image|>\n{prompt}"},
            {"role": "assistant", "content": ""}
        ]
        
        # Process inputs
        inputs = self.processor(messages, images=[image], videos=None)
        inputs.to('cuda')
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens': 100,
            'decode_text': True,
        })
        
        # Generate response
        output = self.model.generate(**inputs)
        return output
    
    def process_output(self, output):
        """Extract score from model output."""
        # TODO: Implement score extraction based on the output format
        # This will depend on how the model formats its response
        return float(output)  # placeholder 