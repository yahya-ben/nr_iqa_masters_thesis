from PIL import Image
import torch
from transformers import TextStreamer, AutoTokenizer
from .base_model import BaseModel


# Usage
# Install
    # Clone this repository and navigate to mPLUG-Owl2 folder
        # git clone https://github.com/X-PLUG/mPLUG-Owl.git
        # cd mPLUG-Owl/mPLUG-Owl2
    # Install Package
        # conda create -n mplug_owl2 python=3.10 -y
        # conda activate mplug_owl2
        # pip install --upgrade pip
        # pip install -e .

# Import required mPLUG-Owl2 specific modules
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class MPLUGOwl2Model(BaseModel):
    def __init__(self, model_config):
        """Initialize mPLUG-Owl2 model."""
        self.model_path = model_config["model_path"]
        
        # Load model and components
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, 
            None, 
            model_name, 
            load_8bit=False, 
            load_4bit=False, 
            device="cuda"
        )
        
        # Initialize conversation template
        self.conv = conv_templates["mplug_owl2"].copy()
        self.roles = self.conv.roles
    
    def generate(self, prompt, image_path):
        """Generate response using mPLUG-Owl2."""
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        
        # Process image
        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        # Format conversation
        inp = DEFAULT_IMAGE_TOKEN + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        
        # Prepare inputs
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)
        
        # Setup stopping criteria
        stop_str = self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # Setup streamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generate response and get embeddings
        with torch.inference_mode():
            # First get the response
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=512,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
            
            # Then get embeddings from forward pass
            outputs = self.model(
                input_ids=output_ids,
                images=image_tensor,
                output_hidden_states=True
            )
            embeddings = outputs.hidden_states[-1]  # Get last hidden state
        
        # Get output text
        response = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        # Reset conversation
        self.conv = conv_templates["mplug_owl2"].copy()
        
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