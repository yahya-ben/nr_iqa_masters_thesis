from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from .base_model import BaseModel
import torch

class LLAVA1_5(BaseModel):
    def __init__(self, model_config):
        """Initialize LLAVA1.5 model."""
        self.model_path = model_config["model_path"]

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Move model to GPU if available
        self.model.to('cuda')
    
    def generate(self, prompt, image_path):
        """Generate response using LLaVA 1.5."""
        # Load and process image
        image = Image.open(image_path)

        conversation = [
            {
        
              "role": "user",
              "content": [
                  {"type": "text", "text": f"{prompt}"},
                  {"type": "image"},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        inputs = self.processor(images=image, text=prompt, return_tensors='pt')

        inputs.to('cuda')
        
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output = self.processor.decode(output[0][2:], skip_special_tokens=True)
       
        return output
    
    def process_output(self, output):
        """Extract score from model output."""
        # TODO: Implement score extraction based on the output format
        # This will depend on how the model formats its response
        return float(output)  # placeholder 