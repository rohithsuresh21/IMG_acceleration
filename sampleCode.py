import torch
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image, LCMScheduler
from PIL import Image
import time

MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"

# ─────────────────────────────────────────
# SESSION STATE — Tracks current image and handles model lazy-loading
# ─────────────────────────────────────────

class GenerationSession:
    """
    Holds the pipelines and current image state so each new prompt
    can build on the previous result.
    """
    def __init__(self):
        self.current_image = None
        self.current_prompt = None
        
        print("Initializing pipelines... (Torch compile may take a minute on first run)")
        
        # This is your base pipeline — txt2img
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,  # FP16 precision
            safety_checker=None,        # remove safety checker to save VRAM + speed
        )
        self.txt2img_pipe.scheduler = LCMScheduler.from_config(self.txt2img_pipe.scheduler.config)
        self.txt2img_pipe.to("cuda")
        self.txt2img_pipe.enable_attention_slicing()
        self.txt2img_pipe.enable_vae_slicing()
        
        # Note: 'reduce-overhead' with fullgraph=True triggers deep C++ AOT compilation.
        # It will feel frozen on the very first generation step while compiling.
        self.txt2img_pipe.unet = torch.compile(
            self.txt2img_pipe.unet,
            mode="reduce-overhead",
            fullgraph=True
        )
        print("txt2img pipeline loaded and compiled.")

        # AutoPipelineForImage2Image shares components from txt2img_pipe (no extra VRAM)
        self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.txt2img_pipe)
        print("img2img pipeline loaded (shared weights).")

    def generate_base(self, prompt: str, negative_prompt: str = "blurry, low quality, distorted"):
        """Full text-to-image generation for the FIRST prompt."""
        start = time.time()
        image = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,      
            guidance_scale=1.0,         
            height=512,
            width=512,
        ).images
        print(f"[txt2img] Generated in {time.time() - start:.2f}s")
        return image

    def generate_variation(
        self,
        prompt: str,
        reference_image: Image.Image,
        strength: float = 0.5,          
        negative_prompt: str = "blurry, low quality, distorted"
    ):
        """Image-to-image generation for INCREMENTAL changes."""
        start = time.time()
        image = self.img2img_pipe(
            prompt=prompt,
            image=reference_image,
            strength=strength,
            num_inference_steps=4,
            guidance_scale=1.0,
            negative_prompt=negative_prompt,
        ).images
        print(f"[img2img] Generated in {time.time() - start:.2f}s")
        return image

    def generate(self, new_prompt: str, strength: float = 0.5):
        if self.current_image is None:
            print("[Session] No previous image, running txt2img")
            self.current_image = self.generate_base(new_prompt)
        else:
            print(f"[Session] Previous image exists, running img2img (strength={strength})")
            self.current_image = self.generate_variation(
                prompt=new_prompt,
                reference_image=self.current_image,
                strength=strength
            )
        self.current_prompt = new_prompt
        return self.current_image

    def reset(self):
        """Start fresh — next generate() call will do full txt2img"""
        self.current_image = None
        self.current_prompt = None
        print("[Session] Reset. Next generation will be full txt2img.")


# ─────────────────────────────────────────
# LOCAL TESTING ONLY (Protected)
# ─────────────────────────────────────────
if __name__ == "__main__":
    # This block only runs if you execute sampleCode.py directly, NOT when imported.
    session = GenerationSession()
    img1 = session.generate("a man sitting on a horse, cinematic lighting, detailed")
    img1.save("step1_horse.png")