"""
Prompt Configuration File
------------------------
Contains versioned prompts and utility functions to access them.
Each prompt can have multiple versions and experiment-specific settings.
"""

prompts = {
    "prompt1": {
        "description": "Basic quality rating prompt",
        "versions": {
            "v1": {
                "text": "Rate the quality of the image.",
                "extraction_method": "direct_output",
                "active": True
            },
            "v2": {
                "text": "Score the quality of the image from 1 to 5, with 1 as lowest and 5 as highest.",
                "extraction_method": "direct_output",
                "active": False
            },
            "v3": {
                "text": "Rate the quality of the image.",
                "extraction_method": "softmax_based",
                "token_pairs": ["good", "poor"],  # for softmax comparison
                "active": True
            }
        }
    },

    "prompt2": {
        "description": "Detailed quality assessment prompt",
        "versions": {
            "v1": {
                "text": """For the given image, please assign a perceptual quality score in terms 
                of structure and texture preservation, color and luminance reproduction, noise, contrast, 
                sharpness, and any other low-level distortions. The score must range from 0 to 100, with a higher 
                score denoting better image quality. Your response must only include a score to summarize its visual quality of the given image. 
                The response format should be: Score: [a score].""",
                "extraction_method": "direct_output",
                "regex_pattern": r"Score:\s*(\d+)",
                "active": True
            },
            "v2": {
                "text": """For the given image, please first detail its perceptual quality in terms of structure and texture preservation, 
                color and luminance reproduction, noise, contrast, sharpness, and any other low-level distortions. Then, based on the perceptual analysis 
                of the given image, assign a quality score to the given image. The score must range from 0 to 100, with a higher score denoting better image quality. 
                Your response must only include a concise description regarding the perceptual quality of the given image, and a score to summarize its perceptual quality of the given image,
                while well aligning with the given description. 
                The response format should be: Description: [a concise description]. Score: [a score].""",
                "extraction_method": "direct_output",
                "regex_pattern": r"Score:\s*(\d+)",
                "description_pattern": r"Description:\s*(.+?)\.\s*Score:",
                "active": True
            }
        }
    },
# Inspiration from Q-Align / Dog-IQA / CCOT (compositional chain-of-thought)

    "prompt3": {
        "description": "Compositional chain-of-thought prompt",
        "type": "chain",
        "chain_order": ["v1", "v2"],
        "versions": {
            "v1": {
                "text": """
                Evaluate the quality of the image as follows: (1) Bad (2) Poor (3) Fair (4) Good (5) Excellent

                For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
                1. Objects that are relevant to answering the question
                2. Object attributes that are relevant to answering the question
                3. Object relationships that are relevant to answering the question

                Scene Graph:
                """,
                "extraction_method": "ccot_direct_guided",
                "requires_json": True,
                "active": True,
                "output_type": "scene_graph"
            },
            "v2": {
                "text": """
                Use the image and following scene graph as context and answer the following question: 
                
                {scene_graph}

                Evaluate the quality of the image as follows: (1) Bad (2) Poor (3) Fair (4) Good (5) Excellent

                Answer with the option's digit from the given choices directly.
                """,
                "extraction_method": "ccot_direct_guided",
                "requires_json": True,
                "active": True,
                "input_type": "scene_graph",
                "output_type": "score"
            }
        }
    },

    "prompt4": {
        "description": "System-role quality assessment prompt",
        "versions": {
            "v1": {
                "text": """You are a helpful assistant to help me evaluate the quality of the image.
                You will be given standards about each quality level. The quality standard is listed as follows:
                5: Excellent, 4: Good, 3: Fair, 2: Bad, 1: Poor.
                Please evaluate the quality of the image and score in [1,2,3,4,5]. Only tell me the number.""",
                "extraction_method": "direct_output",
                "active": True
            }
        }
    },

    "prompt5": {
        "description": "Detailed distortion analysis prompt",
        "versions": {
            "v1": {
                "text": """Based on the image, answer the following questions:

                1. Is the blurriness of the main object of the image noticeable?
                2. Is the blurriness of the background noticeable?
                3. Is the color distortion of the main object of the image noticeable?
                4. Is the color distortion of the background noticeable?
                5. Is the brightness distortion of the main object of the image noticeable?
                6. Is the brightness distortion of the background noticeable?
                7. Is the compression artifact of the main object of the image noticeable?
                8. Is the compression artifact of the background noticeable?
                9. Is the noise of the main object of the image noticeable?
                10. Is the noise of the background noticeable?
                11. Are the spatial distortions of the main object of the image noticeable?
                12. Are the spatial distortions of the background noticeable?""",
                "extraction_method": "direct_output",
                "active": False
            }
        }
    },

    "prompt6": {
        "description": "Distortion identification prompt",
        "versions": {
            "v1": {
                "text": """Based on the image, choose one of these distortions as the most noticeable in the image, 
                and explain why: Blur distortion, Noise distortion, Compression distortion, Color distortion, 
                Brightness distortion, Spatial distortions.""",
                "extraction_method": "direct_output",
                "active": False
            },
            "v2": {
                "text": """What type of distortion is the most prominent in this image.""",
                "extraction_method": "direct_output",
                "active": False
            }
        }
    }
}

def get_active_prompts():
    """Get all active prompts."""
    active_prompts = {}
    for prompt_id, prompt_data in prompts.items():
        for version, config in prompt_data["versions"].items():
            if config.get("active", False):
                active_prompts[f"{prompt_id}_{version}"] = config
    return active_prompts

def get_prompt_version(prompt_id, version):
    """Get a specific prompt version."""
    if prompt_id not in prompts or version not in prompts[prompt_id]["versions"]:
        raise ValueError(f"Prompt {prompt_id} version {version} not found")
    return prompts[prompt_id]["versions"][version]

def get_prompts_by_extraction_method(method):
    """Get all prompts using a specific extraction method."""
    method_prompts = {}
    for prompt_id, prompt_data in prompts.items():
        for version, config in prompt_data["versions"].items():
            if config["extraction_method"] == method and config.get("active", False):
                method_prompts[f"{prompt_id}_{version}"] = config
    return method_prompts 