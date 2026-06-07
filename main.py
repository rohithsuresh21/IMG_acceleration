import gradio as gr
import torch
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = torch.float16 
from LLM_pipeline import smart_generate
from model_loading import GenerationSession
import time

model_id = "runwayml/stable-diffusion-v1-5"
session = GenerationSession(model_id)

def ui_handler(user_prompt):
    start_time = time.time()
    image_list, enhanced_text = smart_generate(user_prompt, session, strength=0.45)
    final_image = image_list if isinstance(image_list, list) else image_list
    end_time = time.time()
    print(f"Image generation time: {end_time:.2f}s")
    
    return final_image, enhanced_text, f"Total generation time: {end_time - start_time:.2f}s"

def ui_reset():
    session.reset()
    return None, "Session cleared. Next generation will be a brand new Base Image.", "Session reset. Next generation will be a brand new Base Image."

with gr.Blocks(title="Active Image Generator", theme=gr.Theme.from_hub("Respair/Shiki")) as demo:
    gr.Markdown("## Active Image Generator\n\nEnter a prompt to generate or modify an image. Each new prompt will build upon the previous image, creating a dynamic and evolving visual experience. Use the reset button to start fresh with a new base image.")
    
    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Describe the image you want to create or modify...")
        generate_button = gr.Button("Generate", variant="primary")
        reset_button = gr.Button("Reset Session", variant="secondary")

    with gr.Column():
        output_image = gr.Image(label="Generated Image")
        enhanced_prompt = gr.Textbox(label="Enhanced Prompt", interactive=False)

    generate_button.click(fn=ui_handler, inputs=prompt_input, outputs=[output_image, enhanced_prompt, gr.Textbox(label="Generation Time", interactive=False)])
    reset_button.click(fn=ui_reset, inputs=None, outputs=[output_image, enhanced_prompt, gr.Textbox(label="Generation Time", interactive=False)])

if __name__ == "__main__":
    demo.launch()
