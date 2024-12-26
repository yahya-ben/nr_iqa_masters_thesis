###### This file contains functions I usually use
###### Most of these functions are written by ChatGPT
###### Some of this code is taken from others code (to be referenced later) 
###### Some of this code is written by me


import pyiqa
import os
from scipy.stats import pearsonr
from scipy import stats

# Needs pyiqa library
# Is called before get_iqa_scores() to init functions
def init_iqa_metrics():
    musiq_metric = pyiqa.create_metric('musiq')
    arniqa_kadid_metric = pyiqa.create_metric('arniqa-kadid')
    topiq_nr_metric = pyiqa.create_metric('topiq_nr')
    tres_metric = pyiqa.create_metric('tres')
    clipiqa_metric = pyiqa.create_metric('clipiqa')
    maniqa_metric = pyiqa.create_metric('maniqa')
    dbcnn_metric = pyiqa.create_metric('dbcnn')
    paq2piq_metric = pyiqa.create_metric('paq2piq')
    hyperiqa_metric = pyiqa.create_metric('hyperiqa')
    cnniqa_metric = pyiqa.create_metric('cnniqa')
    liqe_metric = pyiqa.create_metric('liqe')

# needs pyiqa library
# Returns IQA score in dict format.
# All scores between range between 0 and 100
# Comment out metrics you don't want to use
def get_iqa_scores(image_path):
  return {'musiq': musiq_metric(image_path).cpu().item(),
          'arniqa-kadid': arniqa_kadid_metric(image_path).cpu().item() * 100,
          'topiq_nr': topiq_nr_metric(image_path).cpu().item() * 100,
          'tres': tres_metric(image_path).cpu().item(),
          'clipiqa': clipiqa_metric(image_path).cpu().item() * 100,
          'maniqa': maniqa_metric(image_path).cpu().item() * 100,
          'dbcnn': dbcnn_metric(image_path).cpu().item() * 100,
          'paq2piq': paq2piq_metric(image_path).cpu().item(),
          'hyperiqa': hyperiqa_metric(image_path).cpu().item() * 100,
          'cnniqa': cnniqa_metric(image_path).cpu().item() * 100,
          'liqe' : liqe_metric(image_path).cpu().item(),
        #   'liqe_mix': liqe_mix_metric(image_path).cpu().item(),
        #   'brisque': brisque_metric(image_path).cpu().item(),
        #   'niqe': niqe_metric(image_path).cpu().item(),
        #   'nima': nima_metric(image_path).cpu().item()
          }

# Reference: https://github.com/TianheWu/MLLMs-for-IQA
# Takes an IQA score and scales it to range in scale num.
# Usefull to scale up or down MOS from IQA datasets or IQA evaluators
# Takes a list of scores to be scalled. Scores list is assumed to be from sma IQA dataset or IQA evaluator.
# Returns scaled list of scores.
def scale_values_for_nr(scores, scale_num=5):
    """Scale IQA evaluator scores to match the MOS range (0-5)."""
    min_val = min(scores)
    max_val = max(scores)
    return [(score - min_val) / (max_val - min_val) * scale_num for score in scores]


def get_image_mos(image_name, mos_file_path):
    """
    Retrieves the MOS (Mean Opinion Score) for a given image name from a specified text file.

    Parameters:
    - image_name: Name of the image file (e.g., 'I81_20_05.png').
    - mos_file_path: Path to the text file containing image names and MOS values.

    Returns:
    - The MOS value for the given image, or None if the image is not found.
    """
    mos_dict = {}

    # Step 1: Open the MOS file and read line by line
    with open(mos_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, mos_value = parts
                mos_dict[img_name] = float(mos_value)

    # Step 2: Retrieve the MOS value for the specified image name
    return mos_dict.get(image_name, None)


# When you have a folder containing images and want to load it
# Needs os library
# Returns list with images in it
def load_images_folder(images_folder_path, sort=False):
    
    images = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if sort:
        images.sort()
    
    return images


# Calculates PRCC and SRCC using scipy library
# Could take python list and pandas series.
def calculate_prcc_srcc(model_pred, mos_list):
    prcc, _ = pearsonr(model_pred, mos_list)
    srcc, _ = stats.spearmanr(model_pred, mos_list)

    return prcc, srcc


# Usuall function that takes a bunch of images and evaluates their quality
def iqa_test(images_folder_path, with_print=False):
    # Get list of images from folder
    images = load_images_folder(images_folder_path)
    
    # Dictionary to store results for all images
    all_scores = {}
    
    # Process each image
    for image in images:
        image_path = os.path.join(images_folder_path, image)
        scores_image = get_iqa_scores(image_path)
        all_scores[image] = scores_image
        
        # Optional: Print progress
        if with_print:
            print(f"Processed {image}: {scores_image}")
    
    return all_scores
