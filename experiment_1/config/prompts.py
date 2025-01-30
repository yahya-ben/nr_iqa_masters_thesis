# you add new prompts, but not modify these.
prompts = {
    "prompt1": {
        "text": "Rate the quality of the image.",
        "extraction_method": "direct_output"
    },
    "prompt2": {
        "text": "Score the quality of the image from 1 to 5, with 1 as lowest and 5 as highest.",
        "extraction_method": "direct_output"
    },
#     "prompt3": {
#         "text": "Rate the quality of the image <image>.",
#         "extraction_method": "softmax_based"
#     },
#     "prompt4": {
#         "text": "Rate the quality of the image <image>.",
#         "extraction_method": "prompt_ensemble"
#     },

    "prompt5": {
        "text": """For the given image, please assign a perceptual quality score in terms 
        of structure and texture preservation, color and luminance reproduction, noise, contrast, 
        sharpness, and any other low-level distortions. The score must range from 0 to 100, with a higher 
        score denoting better image quality. Your response must only include a score to summarize its visual quality of the given image. 
        The response format should be: Score: [a score].""",
        "extraction_method": "direct_output"
    },

    "prompt6": {
        "text": """For the given image, please first detail its perceptual quality in terms of structure and texture preservation, 
        color and luminance reproduction, noise, contrast, sharpness, and any other low-level distortions. 
        Then, based on the perceptual analysis of the given image, assign a quality score to the given image. 
        The score must range from 0 to 100, with a higher score denoting better image quality.""",
        "extraction_method": "direct_output"
    },
    
    "prompt7": {
        "text": """System: You are a helpful assistant to help me evaluate the quality of the image.
You will be given standards about each quality level. The quality standard is listed as follows:
5: Excellent, 4: Good, 3: Fair, 2: Bad, 1: Poor.
User: please evaluate the quality of the image and score in [1,2,3,4,5]. Only tell me the number.""",
        "extraction_method": "direct_output"
    }
} 