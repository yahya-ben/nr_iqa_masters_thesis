from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
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
        
        # Move model to GPU if available
        self.model.to('cuda')
    
    def generate(self, prompt, image_path):
        """Generate response using LLaVA 1.5."""
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

        # Apply the chat template to obtain the full prompt text
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process the image and text to obtain the inputs for the model
        inputs = self.processor(images=image, text=prompt_text, return_tensors='pt')
        
        # Move all tensors to the GPU
        for key in inputs:
            inputs[key] = inputs[key].to('cuda')
        
        # Generate the output using the model
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

        # For now, we assume that 'output' carries the embeddings you need.
        embeds = output  # saving the embeddings
        
        # Decode the generated tokens (skipping the first two tokens)
        decoded_output = self.processor.decode(output[0][2:], skip_special_tokens=True)
        
        return decoded_output, embeds
       
    # this code snippet follows Q-Bench
    def process_output(self, embeds):
        """Extract score from model output."""

        print("heyyyyyyyyyyyy")
        
        tokenized = self.processor.tokenizer(["good", "poor"], add_special_tokens=False)
        # Assuming each tokenized word returns a list of token ids and we want the first token
        good_idx = tokenized["input_ids"][0][0]
        poor_idx = tokenized["input_ids"][1][0]

        print(f"good {good_idx}\n")
        print(f"poor {poor_idx}\n")

        
        output_logits = embeds[0,-1]

        print(f"output_logits: {output_logits}")
        
        # # Pass the embeddings through the model to get logits.
        # # This assumes your model accepts 'input_embeds'
        # model_outputs = self.model(input_embeds=embeds)
        # output_logits = model_outputs.logits  # This is the logits tensor
        
        # For example, if output_logits has shape (batch_size, sequence_length, vocab_size)
        # and you want the logits of the last token in the sequence for the first sample:
        
        # Extract logits corresponding to "good" and "poor"
        selected_logits = output_logits[[good_idx, poor_idx]]
        
        # Apply softmax after scaling if needed
        q_pred = (selected_logits / 100).softmax(dim=0)[0]
        
        return float(q_pred)
