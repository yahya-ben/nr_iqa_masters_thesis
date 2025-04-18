import os
import pandas as pd
import json
import re
from datetime import datetime
from config.prompts import prompts, get_active_prompts
from config.datasets import datasets
from config.models import models
from models import load_model

def load_images_folder(images_folder_path, sort=False):
    """Load images from folder."""
    images = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if sort:
        images.sort()
    return images

def process_direct_output_with_regex(raw_prediction, regex_pattern):
    """Process outputs that need specific regex extraction."""
    try:
        match = re.search(regex_pattern, raw_prediction)
        if match:
            return float(match.group(1))
    except:
        return None
    return raw_prediction

def process_softmax_based(embeddings, token_pairs):
    """Process softmax-based outputs."""
    # Note: This is already handled in the model's process_output method
    return embeddings

def process_ccot_direct_guided(raw_prediction):
    """Process guided outputs (like scene graphs)."""
    try:
        # Try to extract JSON content
        json_match = re.search(r'\{.*\}', raw_prediction, re.DOTALL)
        if json_match:
            scene_graph = json.loads(json_match.group())
            # Process scene graph to derive a score
            # This is a placeholder - implement your scoring logic
            return 0.0
    except:
        return None
    return None

def process_scene_graph_prompt_output(scene_graph):
    """Extracts the JSON scene graph from the provided text."""
    
    # Split the text to find the JSON part after "ASSISTANT: "
    parts = scene_graph.split("ASSISTANT: ")
    # if len(parts) < 2:
    #     return None  # No JSON found
    
    json_str = parts[-1].strip()  # Take the last part and remove whitespace
    return json_str
    

def process_ccot_prompt(followup_prompt, scene_graph, model_instance, image_path):
    """Process prompts that work in a chain."""

    sg = process_scene_graph_prompt_output(scene_graph)

    # print(f"Scene graph only: {scene_graph}")
    
    followup_prompt = followup_prompt.format(scene_graph=sg)

    # print(f"Followup prompt: {followup_prompt}")
        
    # Get model prediction
    raw_prediction, embeddings = model_instance.generate(followup_prompt, image_path)

    # score = process_ccot_direct_guided(raw_prediction)
    score = raw_prediction
    return score

def save_results_to_csv(results, model_name, timestamp):
    """Save results to CSV file for a specific model."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare data for DataFrame
    rows = []
    model_data = results[model_name]
    
    for dataset, dataset_data in model_data.items():
        for prompt, predictions in dataset_data.items():
            for image_id, score in predictions.items():
                row = {
                    'model_name': models[model_name]['model_name'],  # Use actual model name
                    'dataset': dataset,
                    'prompt': prompt,
                    'image_id': image_id,
                    'predicted_score': score
                }
                rows.append(row)
    
    # Create the filename with experiment number
    filename = f"{models[model_name]['model_name']}_results_{timestamp}.csv"
    csv_path = os.path.join(results_dir, filename)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return df

def run_experiment():
    """Main function to run the experiment."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Starting experiment")
    results = {}
    
    # Get active prompts
    active_prompts = get_active_prompts()
    print(f"Active prompts: {list(active_prompts.keys())}")
    
    for model, model_config in models.items():
        print(f"Processing model: {model} ({model_config['model_name']})")
        results[model] = {}
        
        # Initialize model
        model_instance = load_model(model_config)
        
        for dataset_name, dataset_config in datasets.items():
            print(f"Processing dataset: {dataset_name}")
            results[model][dataset_name] = {}
            
            try:
                # Load dataset samples
                dataset_path = dataset_config["path"]
                print(f"Dataset path: {dataset_path}")
                
                sample_size = dataset_config["sample_size"]
                print(f"Sample size: {sample_size}")
                
                # TODO: Better Dataset loading (batching using Dataloader ig)
                
                samples = load_images_folder(dataset_path, sort=True)
                print(f"Loaded {len(samples)} samples from {dataset_name}")
                
                for prompt_name, prompt_config in active_prompts.items():
                    print(f"Applying prompt: {prompt_name}")
                    predictions = {}
                    
                    try:
                        for i, image_id in enumerate(samples):
                            print(f"Processing image: {image_id}")
                            try:
                                image_path = os.path.join(dataset_path, image_id)
                                
                                
                                if prompt_name == "prompt3_v1":
                                    scene_graph, embeddings = model_instance.generate(prompt_config["text"], image_path)
                                    # print(scene_graph)
                                    score = -1
                                elif prompt_name == "prompt3_v2":
                                    score = process_ccot_prompt(prompt_config["text"], scene_graph, model_instance, image_path)
                                
                                else:
                                    # Regular prompt processing
                                    raw_prediction, embeddings = model_instance.generate(prompt_config["text"], image_path)
                                    
                                    # Process the output based on extraction method
                                    if prompt_config["extraction_method"] == "direct_output":
                                        if "regex_pattern" in prompt_config:
                                            score = process_direct_output_with_regex(
                                                raw_prediction, 
                                                prompt_config["regex_pattern"]
                                            )
                                        else:
                                            score = raw_prediction
                                    
                                    elif prompt_config["extraction_method"] == "softmax_based":
                                        score = model_instance.process_output(embeddings)
                                        score = -1
                                    
                                    elif prompt_config["extraction_method"] == "ccot_direct_guided":
                                        score = process_ccot_direct_guided(raw_prediction)
                                    
                                    else:
                                        print(f"Unknown extraction method: {prompt_config['extraction_method']}")
                                        score = None

                                predictions[image_id] = score
                                
                                if (i + 1) % 10 == 0:
                                    print(f"Processed {i + 1}/{len(samples)} samples")
                                    
                            except Exception as e:
                                print(f"Error processing sample {image_id}: {str(e)}")
                                predictions[image_id] = None
                        
                        results[model][dataset_name][prompt_name] = predictions
                            
                    except Exception as e:
                        print(f"Error processing prompt {prompt_name}: {str(e)}")
                        results[model][dataset_name][prompt_name] = {"error": str(e)}
                        
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                results[model][dataset_name] = {"error": str(e)}
        
        # Save results for this model
        df = save_results_to_csv(results, model, timestamp)
    
    print("Experiment completed")
    return results

if __name__ == "__main__":
    results = run_experiment()