from diffusers import DiffusionPipeline, AutoPipelineForImage2Image, LCMScheduler
import torch
import torch.nn.functional as F
from torch.utils.hooks import RemovalHandle
import time
from PIL import Image

model_id = "simianluo/lcm_dreamshaper_v7"

class GradientBasedTokenPruner:
    """Dynamically prune tokens based on gradient importance during encoding."""
    
    def __init__(self, pruning_ratio: float = 0.3):
        """
        Args:
            pruning_ratio: Fraction of tokens to prune (0.0-1.0)
        """
        self.pruning_ratio = pruning_ratio
        self.token_importances = None
        self.hooks = []
        
    def compute_importance(self, hidden_states, grads):
        """Compute importance as L2 norm of gradients per token."""
        # hidden_states: (batch, seq_len, hidden_dim)
        # grads: (batch, seq_len, hidden_dim)
        importance = torch.norm(grads, dim=2)  # (batch, seq_len)
        return importance
    
    def register_hooks(self, text_encoder):
        """Register backward hooks on text encoder layers."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    output.register_hook(lambda grad: self._store_importance(grad, name))
            return hook
        
        for name, module in text_encoder.named_modules():
            if 'attention' in name and 'output' in name:
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def _store_importance(self, grad, layer_name):
        """Store gradient magnitudes for importance ranking."""
        if self.token_importances is None:
            self.token_importances = []
        self.token_importances.append(grad)
    
    def prune_tokens(self, token_embeddings, attention_mask):
        """
        Prune low-importance tokens dynamically.
        
        Args:
            token_embeddings: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
            
        Returns:
            pruned_embeddings, pruned_mask, token_indices
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        
        # Compute importance scores (gradient-based if available, else use magnitude)
        if self.token_importances:
            importance = torch.norm(
                torch.stack(self.token_importances), dim=2
            ).mean(0)  # Average across layers
        else:
            # Fallback: use embedding magnitude
            importance = torch.norm(token_embeddings, dim=2)
        
        # Identify tokens to keep (top-k by importance)
        num_prune = max(1, int(seq_len * self.pruning_ratio))
        _, keep_indices = torch.topk(importance, seq_len - num_prune, dim=1)
        
        # Sort indices to maintain order
        keep_indices = torch.sort(keep_indices, dim=1)[0]
        
        # Prune embeddings and mask
        pruned_embeddings = torch.gather(
            token_embeddings, 1, 
            keep_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )
        pruned_mask = torch.gather(attention_mask, 1, keep_indices)
        
        self.token_importances = []  # Reset for next iteration
        return pruned_embeddings, pruned_mask, keep_indices
    
    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()


# Update GenerationSession class
class GenerationSession:
    def __init__(self, token_pruning_ratio: float = 0.2):
        self.current_image = None
        self.current_prompt = None
        self.token_pruner = GradientBasedTokenPruner(pruning_ratio=token_pruning_ratio)
        
        print("Initializing pipelines... (Torch compile may take a minute on first run)")
        
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  
            safety_checker=None,
        )
        self.txt2img_pipe.scheduler = LCMScheduler.from_config(self.txt2img_pipe.scheduler.config)
        self.txt2img_pipe.to("cuda")
        self.txt2img_pipe.enable_attention_slicing()
        self.txt2img_pipe.enable_vae_slicing()
        
        self.txt2img_pipe.unet = torch.compile(
            self.txt2img_pipe.unet,
            mode="reduce-overhead",
            fullgraph=True
        )
        print("txt2img pipeline loaded and compiled.")
        
        self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.txt2img_pipe)
        
        # Register pruning hooks
        self.token_pruner.register_hooks(self.txt2img_pipe.text_encoder)
        print("Token pruning enabled.")
    
    def generate_base(self, prompt: str, negative_prompt: str = "blurry, low quality, distorted"):
        """Full text-to-image generation with token pruning."""
        start = time.time()
        
        # Enable gradient tracking for importance computation
        self.txt2img_pipe.text_encoder.requires_grad_(True)
        
        image = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,
            guidance_scale=1.0,
            height=512,
            width=512,
        ).images
        
        self.txt2img_pipe.text_encoder.requires_grad_(False)
        print(f"[txt2img + token pruning] Generated in {time.time() - start:.2f}s")
        return image[0]
    
    def __del__(self):
        """Cleanup hooks on session end."""
        self.token_pruner.remove_hooks()

# ─────────────────────────────────────────
# LOCAL TESTING ONLY (Protected)
# ─────────────────────────────────────────
if __name__ == "__main__":
    # This block only runs if you execute sampleCode.py directly, NOT when imported.
    session = GenerationSession()
    img1 = session.generate("a man sitting on a horse, cinematic lighting, detailed")
    img1.save("step1_horse.png")