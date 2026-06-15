# MODELS = {
#     "Gemma-3-4B":          "google/gemma-3-4b-it",
#     "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
#     "OLMo-2-7B":           "allenai/OLMo-2-1124-7B-Instruct",
#     "Qwen3-8B":            "Qwen/Qwen3-8B",
#     "ModernBERT-8B":       "answerdotai/ModernBERT-large",
#     "Llama-3-8B":          "meta-llama/Llama-3.1-8B"
# }

MODELS = {
    "Gemma-3-1B": "google/gemma-3-1b-it"
}

PROMPT_FILES = {
    "general": "general_prompt.yaml",
    #"rsa": "rsa_prompt.yaml",
    #"reasoning": "reasoning_prompt.yaml"
}

SEEDS = {
    "Gemma-3-1B"         : 1,
    "Gemma-3-4B"         : 42,
    "Mistral-7B-Instruct": 77,
    "OLMo-2-7B"          : 123,
    "Qwen3-8B"           : 256,
    "ModernBERT-8B"      : 999,
    "Llama-3-8B"         : 1337,
}