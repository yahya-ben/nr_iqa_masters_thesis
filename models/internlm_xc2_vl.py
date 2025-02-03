from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseModel
import torch

class InternLMXC2Model(BaseModel):
    def __init__(self, model_config):
        """Initialize InternLMXC2Model model."""
        self.model_path = model_config["model_path"]
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float32, trust_remote_code=True).cuda()
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
        
        self.model = self.model.eval()
    
    def generate(self, prompt, image_path):
        """Generate response using InternLMXC2Model."""
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Format messages
        image = self.model.vis_processor(image)

        image = torch.stack(image)
        
        query = f'<ImageHere> <ImageHere>{prompt}'
        with torch.cuda.amp.autocast():
            # Get both response and embeddings
            response, history = self.model.chat(
                self.tokenizer, 
                query=query, 
                image=image, 
                history=[], 
                do_sample=False,
                return_embeddings=True  # Add this parameter if available in the model
            )
            
            # If the model doesn't have a return_embeddings parameter,
            # you can get embeddings directly from the forward pass:
            inputs = self.tokenizer(query, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1]  # Get last hidden state
        
        return response, embeddings
    
    def process_output(self, embeddings):
        """Extract score from model output."""
        # Get token IDs for "good" and "poor"
        good_idx, poor_idx = self.tokenizer(["good", "poor"]).input_ids
        
        # Get logits from the last position
        output_logits = self.model(input_embeds=embeddings).logits[0, -1]
        
        # Calculate prediction score
        q_pred = (output_logits[[good_idx, poor_idx]] / 100).softmax(0)[0]
        
        return float(q_pred) 