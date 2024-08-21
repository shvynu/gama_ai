import os
import sys
import subprocess
import torch
import numpy as np
from PIL import Image
import random
import streamlit as st

# Clone the repository if it doesn't exist
if not os.path.exists('ComfyUI'):
    subprocess.run(['git', 'clone', '-b', 'totoro3', 'https://github.com/camenduru/ComfyUI.git'])

# Ensure the necessary modules are in the path
sys.path.append(os.path.abspath('ComfyUI'))

# Now we can import the required modules
from nodes import NODE_CLASS_MAPPINGS
from totoro_extras import nodes_custom_sampler
from totoro import model_management

# Force CPU usage
torch.cuda.is_available = lambda : False
model_management.get_torch_device = lambda: torch.device('cpu')
model_management.get_total_memory = lambda *args, **kwargs: 8 * 1024 * 1024 * 1024  # Assume 8GB of RAM

# Initialize TotoroUI nodes
DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# Function to find the closest number divisible by m
def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    return n1 if abs(n - n1) < abs(n - n2) else n2

# Streamlit app
st.title("TotoroUI Image Generator")
st.write("Generate images using a TotoroUI-powered model based on your input prompts.")

# Input fields
positive_prompt = st.text_input("Enter a prompt for the image", "man playing PS4 in a gaming arena")
width = st.slider("Image Width (px)", min_value=256, max_value=512, value=384, step=16)
height = st.slider("Image Height (px)", min_value=256, max_value=512, value=384, step=16)
steps = st.slider("Sampling Steps", min_value=1, max_value=30, value=15)
sampler_name = st.selectbox("Sampling Method", ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "ddim"])
scheduler = st.selectbox("Scheduler", ["normal", "karras", "exponential", "simple"])
seed = st.number_input("Random Seed", min_value=0, max_value=18446744073709551615, value=0)

# Generate Image button
if st.button("Generate Image"):
    if seed == 0:
        seed = random.randint(0, 18446744073709551615)
    st.write(f"Seed: {seed}")
    
    with st.spinner("Generating image... This may take several minutes."):
        try:
            with torch.inference_mode():
                # Load models
                clip = DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
                unet = UNETLoader.load_unet("flux1-schnell.safetensors", "fp8_e4m3fn")[0]
                vae = VAELoader.load_vae("ae.sft")[0]
                
                # Generate image
                cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
                cond = [[cond, {"pooled_output": pooled}]]
                noise = RandomNoise.get_noise(seed)[0]
                guider = BasicGuider.get_guider(unet, cond)[0]
                sampler = KSamplerSelect.get_sampler(sampler_name)[0]
                sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]
                latent_image = EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
                sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
                model_management.soft_empty_cache()
                decoded = VAEDecode.decode(vae, sample)[0].detach()
                
                # Save and display the generated image
                img = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
                st.image(img, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.write("Note: This application is running on CPU, which results in slower performance. Image generation may take several minutes.")
