from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from .base_model import BaseModel
import torch

class LLAVA1_6(BaseModel):
    def __init__(self, model_config):

        self.model_path = model_config["model_path"]

        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.float16) 
                
        # Move model to GPU if available
        self.model.to('cuda')
    
    def generate(self, prompt, image_path):

        # Load and process image
        image = Image.open(image_path).convert("RGB")

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

        output = self.model.generate(**inputs, max_new_tokens=300)
        embeds = output # getting the embeddings out
        output = self.processor.decode(output[0], skip_special_tokens=True)
       
        return output, embeds
    
  # this code snippet follows Q-Bench
    def process_output(self, embeds):
        """Extract score from model output."""

        good_idx, poor_idx = self.processor.tokenizer(["good","poor"]).tolist()

        output_logits = self.model(input_embeds=embeds).logits[0,-1]

        q_pred = (output_logits[[good_idx, poor_idx]] / 100).softmax(0)[0]
        
        return float(q_pred) 