import streamlit as st
from PIL import Image, ImageOps
import torch
import einops
from stable_diffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import math
from torch import autocast
import numpy as np
from einops import rearrange
import k_diffusion as K
import sys

sys.path.append("./stable_diffusion")

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


# Load and display image
def load_image(image_file):
    img = Image.open(image_file)
    global input_image
    input_image = Image.open(image_file).convert("RGB")
    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
    return img

# Generate styled image
def generate_style_image(model, style):
    prompt = f"turn the person/object in this image to {style} character"
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([prompt])]
        global input_image
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(100)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": 7.5,
            "image_cfg_scale": 1.5,
        }
        torch.manual_seed(0)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    return edited_image

# Main function to run the app
def main():
    _, _, col3, _ = st.columns([1, 1, 4, 1])
    col3.title("FlickFantasy")

    # Upload button
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Style dropdown
    style = st.selectbox("Choose Style", ["Avatar", "Harry Potter", "Pirates of the Carribean", "Lord of the Rings", "Star Wars", "Marvel"])

    # Custom prompt button
    if st.button("Custom Prompt"):
        custom_prompt = st.text_area("Enter your custom prompt")
        if custom_prompt:
            # Here you can implement the logic to handle the custom prompt
            st.success("Custom prompt sent successfully!")

    # Central alignment for buttons
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    # State variable to track button click
    magic_button_clicked = col3.button("Magic 🪄")

    # Display image section
    if uploaded_file is not None:
        # Load and display uploaded image
        original_image = load_image(uploaded_file)
        col4, col5 = st.columns(2)
        col4.header("Original Image")
        col4.image(original_image, use_column_width=True)

        # Generate and display styled image
        if magic_button_clicked:
            styled_image = generate_style_image(original_image, style)
            col5.header("Styled Image")
            col5.image(styled_image, use_column_width=True)
    elif uploaded_file is None:
        st.warning("Please upload an image.")
    elif not magic_button_clicked:
        st.info("Click 'Magic 🪄' to generate styled image.")

if __name__ == "__main__":
    config = OmegaConf.load("configs/generate.yaml")
    model = load_model_from_config(config, "checkpoints/instruct-pix2pix-00-22000.ckpt", None)
    model.eval().cuda()
    print("<-------------------Model Checkpoint Loaded---------------->")
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    print("<-------------------Config---------------->")
    null_token = model.get_learned_conditioning([""])
    print("<-------------------Conditioning Learnt---------------->")
    main()