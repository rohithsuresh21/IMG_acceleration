import os
# Prevent network telemetry hanging during initial setups
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import requests
from sampleCode import GenerationSession

def enhance_prompt(user_prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"[INST] You are an image generation prompt engineer. "
                          f"Rewrite this prompt to be vivid and detailed, under 60 words. "
                          f"Return ONLY the rewritten prompt, nothing else.\n\n"
                          f"Prompt: {user_prompt} [/INST]",
                "stream": False
            },
            timeout=15
        )
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        print("Warning: Could not connect to local Ollama instance at localhost:11434. Using raw prompt.")
        return user_prompt


def smart_generate(user_prompt: str, session: GenerationSession, strength: float = 0.45):
   
    enhanced = enhance_prompt(user_prompt)
    print(f"\nOriginal : {user_prompt}")
    print(f"Enhanced : {enhanced}")

    image = session.generate(enhanced, strength=strength)
    return image, enhanced


# ─────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Create one shared session containing the models
    session = GenerationSession()

    # Generation Sequence
    img1, p1 = smart_generate("a man sitting on a horse", session)
    img1.save("step1.png")

    img2, p2 = smart_generate("add a space suit on the man", session, strength=0.45)
    img2.save("step2.png")

    img3, p3 = smart_generate("give the visor a golden color", session, strength=0.35)
    img3.save("step3.png")
    
    print("\nAll pipeline steps completed successfully!")