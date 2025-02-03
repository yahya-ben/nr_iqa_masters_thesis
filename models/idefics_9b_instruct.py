from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor
from .base_model import BaseModel
import torch

class IDEFICS9bModel(BaseModel):
    def __init__(self, model_config):
        """Initialize IDEFICS 9B Instruct model."""
        self.model_path = model_config["model_path"]

        # Load model and tokenizer
        self.model = IdeficsForVisionText2Text.from_pretrained(self.model_path, torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Move model to GPU if available
        self.model.to('cuda')
    
    def generate(self, prompt, image_path):
        """Generate response using IDEFICS 9B Instruct."""
        # Load and process image
        image = Image.open(image_path).convert("RGB")

        prompts = ["user:",image,f"{prompt}","Assistant:"]
        
        inputs = self.processor(prompts,return_tensors="pt",debug=True).to('cuda')

        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        
        generate_ids = self.model.generate(**inputs,
                                      eos_token_id=exit_condition,
                                      bad_words_ids=bad_words_ids,
                                      max_length=400)
        
        embeds = generate_ids
    
        generate_text = self.processor.batch_decode(generate_ids,
                                               skip_special_tokens=True)[0]
        
        return generate_text, embeds
        
            
    def process_output(self, embeds):
        """Extract score from model output."""

        good_idx, poor_idx = self.processor.tokenizer(["good","poor"]).tolist()

        output_logits = self.model(input_embeds=embeds).logits[0,-1]

        q_pred = (output_logits[[good_idx, poor_idx]] / 100).softmax(0)[0]
        
        return float(q_pred) 