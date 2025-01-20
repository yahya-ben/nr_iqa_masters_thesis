import os
import pandas as pd
from datetime import datetime
from config.prompts import prompts
from config.datasets import datasets
from config.models import models
from models import load_model

def save_results_to_csv(results, timestamp):
    """Save results to CSV file"""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare data for DataFrame
    rows = []
    for model, model_data in results.items():
        for dataset, dataset_data in model_data.items():
            for prompt, predictions in dataset_data.items():
                for image_id, score in predictions.items():
                    row = {
                        'model': model,
                        'dataset': dataset,
                        'prompt': prompt,
                        'image_id': image_id,
                        'predicted_score': score
                    }
                    rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    csv_path = f'{results_dir}/results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return df

def run_experiment():
    """Main function to run the experiment."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("Starting experiment")
    results = {}
    
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
                sample_size = dataset_config["sample_size"]
                
                # TODO: Implement dataset loading
                # samples = load_dataset(dataset_path, sample_size)
                # Format: [(image_id, image_path), ...]
                samples = []  # placeholder
                
                print(f"Loaded {len(samples)} samples from {dataset_name}")
                
                for prompt_name, prompt_config in prompts.items():
                    print(f"Applying prompt: {prompt_name}")
                    predictions = {}
                    
                    try:
                        for i, (image_id, image_path) in enumerate(samples):
                            try:
                                # Format prompt with image
                                formatted_prompt = prompt_config["text"].replace("<image>", image_path)
                                
                                # Get model prediction
                                raw_prediction = model_instance.generate(formatted_prompt, image_path)
                                score = model_instance.process_output(raw_prediction)
                                
                                predictions[image_id] = score
                                
                                if (i + 1) % 10 == 0:  # Print progress every 10 samples
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
    
    # Save results to CSV
    df = save_results_to_csv(results, timestamp)
    print("Experiment completed")
    
    return results, df

if __name__ == "__main__":
    results, df = run_experiment()