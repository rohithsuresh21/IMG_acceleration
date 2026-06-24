from PIL import Image
import os
import ollama
import torch
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image, LCMScheduler, AutoPipelineForText2Image
import time

model_id = "simianluo/lcm_dreamshaper_v7"

class GenerationSession:
    def __init__(self, model_id):
        self.model_id = model_id   
        self.txt2img_pipeline = None  
        self.img2img_pipeline = None
        self.current_image = None
        self.current_prompt = None
        self._initialize_pipelines()

    def _initialize_pipelines(self):
            print("initializing pipelines...")

            self.txt2img_pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype = torch.float16,
                safety_checker = None
            )
            
            self.txt2img_pipeline.scheduler = LCMScheduler.from_config(self.txt2img_pipeline.scheduler.config)
            self.txt2img_pipeline.to("cuda")
            self.txt2img_pipeline.enable_attention_slicing()
            self.txt2img_pipeline.enable_vae_slicing()

           
            print("Text 2 image pipeline loaded and compiled.")


            self.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(self.txt2img_pipeline)  
            print("Image 2 image pipeline loaded (shared weights).")

    def GeneratingBaseImage(self, prompt: str, negative_prompt: str = "Blurry, low quality, static and distorted image") -> str:
        start = time.time()
        image = self.txt2img_pipeline(
            prompt = prompt,
            negative_prompt= negative_prompt,
            num_inference_steps = 4,
            guidance_scale = 1.0,
            height = 512,
            width = 512
        ).images
        print(f"Text to image generated in [{time.time() - start:.2f}s]")
        return image
    
    def GeneratingVariationImage(self, prompt: str, reference_image: Image.Image, strength: float = 0.5, negative_prompt: str = "Blurry, low quality, static and distorted image") -> str:
        start = time.time()
        image = self.img2img_pipeline(
            prompt = prompt,
            image = reference_image,
            strength = strength,
            num_inference_steps = 4,
            guidance_scale = 1.0,
            negative_prompt = negative_prompt
        ).images
        print(f"Image to image generated in [{time.time() - start:.2f}s]")
        return image
    
    def Generate(self, new_prompt: str, strength: float = 0.5):
        if self.current_image is None:
            self.current_image = self.GeneratingBaseImage(new_prompt)
        else:
            self.current_image = self.GeneratingVariationImage(new_prompt, self.current_image, strength)
        
        self.current_prompt = new_prompt
        return self.current_image
    
    def reset(self):
        self.current_image = None
        self.current_prompt = None
        print("Session reset. Ready for new generation.")

