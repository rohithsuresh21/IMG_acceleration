import os
import requests
from model_loading import GenerationSession

def prompt_enhancer(user_prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "format": "json",  
                "prompt": (
                    f"[INST] You are an image generation prompt engineer. "
                    f"Rewrite this prompt to be vivid and detailed, under 60 words. "
                    f"Return ONLY the rewritten prompt, nothing else.\n\n"
                    f"Prompt: {user_prompt} [/INST]"
                ),
                "stream": False
            },
            timeout=15 
        )
    except requests.exceptions.ConnectionError:
        print("Warning: Could not connect to local Ollama.")
        return user_prompt
    return response.json()["response"].strip()

def smart_generate(user_prompt: str, session: GenerationSession, strength: float = 0.45):
    enhanced = prompt_enhancer(user_prompt)

    image = session.generate(enhanced, session.current_image, strength=strength)
    return image, enhanced
