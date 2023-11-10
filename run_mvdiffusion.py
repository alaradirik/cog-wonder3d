import os
import random
from packaging import version
from collections import defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from omegaconf import OmegaConf

import cv2
import xformers
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from rembg import remove
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from utils.misc import load_config
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path:str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation
    cond_on_normals: bool
    cond_on_colors: bool


def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


class MultiViewDiffusion:
    def __init__(self):
        self.cfg = load_config("configs/mvdiffusion-joint-ortho-6views.yaml")
        schema = OmegaConf.structured(TestConfig)
        self.cfg = OmegaConf.merge(schema, self.cfg)

        # Load scheduler, tokenizer and models.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="image_encoder", revision=self.cfg.revision
        )
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="feature_extractor", revision=self.cfg.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="vae", revision=self.cfg.revision
        )
        self.unet = UNetMV2DConditionModel.from_pretrained_2d(
            self.cfg.pretrained_unet_path, 
            subfolder="unet", 
            revision=self.cfg.revision, 
            **self.cfg.unet_from_pretrained_kwargs
        )
        self.unet.enable_xformers_memory_efficient_attention()

        device = 'cuda'
        weight_dtype = torch.float32
        self.image_encoder.to(device, dtype=torch.float32)
        self.vae.to(device, dtype=torch.float32)
        self.unet.to(device, dtype=torch.float32)

        scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.pipeline = MVDiffusionImagePipeline(
            image_encoder=self.image_encoder, 
            feature_extractor=self.feature_extractor, 
            vae=self.vae, unet=self.unet, safety_checker=None,
            scheduler=scheduler,
            **self.cfg.pipe_kwargs
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def generate_views(self, seed=None, weight_dtype=torch.float32):
        name = 'validation'
        save_dir = self.cfg.save_dir

        # Get the  dataset
        validation_dataset = MVDiffusionDataset(**self.cfg.validation_dataset)

        # DataLoaders creation:
        dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=self.cfg.validation_batch_size, 
            shuffle=False, 
            num_workers=self.cfg.dataloader_num_workers
        )

        os.makedirs(self.cfg.save_dir, exist_ok=True)

        # If passed along, set the training seed now.
        if seed is None:
            generator = None
            seed = random.randint(0, 10000)
        else:
            self.cfg.seed = seed
            set_seed(self.cfg.seed)
            generator = torch.Generator(device="cuda").manual_seed(self.cfg.seed)
        
        images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)
        for i, batch in tqdm(enumerate(dataloader)):
            # repeat  (2B, Nv, 3, H, W)
            imgs_in = torch.cat([batch['imgs_in']]*2, dim=0)
            filename = batch['filename']
            
            # (2B, Nv, Nce)
            camera_embeddings = torch.cat([batch['camera_embeddings']]*2, dim=0)
            task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)
            camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

            imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
            camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

            images_cond.append(imgs_in)
            num_views = 6
            VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

            with torch.autocast("cuda"):
                for guidance_scale in self.cfg.validation_guidance_scales:
                    out = self.pipeline(
                        imgs_in, 
                        camera_embeddings, 
                        generator=generator, 
                        guidance_scale=guidance_scale, 
                        output_type='pt', 
                        num_images_per_prompt=1, 
                        **self.cfg.pipe_validation_kwargs
                    ).images

                    bsz = out.shape[0] // 2
                    normals_pred = out[:bsz]
                    images_pred = out[bsz:]

                    cur_dir = save_dir

                    for i in range(bsz//num_views):
                        scene = filename[i]
                        scene_dir = os.path.join(cur_dir, scene)
                        normal_dir = os.path.join(scene_dir, "normals")
                        masked_colors_dir = os.path.join(scene_dir, "masked_colors")

                        os.makedirs(normal_dir, exist_ok=True)
                        os.makedirs(masked_colors_dir, exist_ok=True)

                        for j in range(num_views):
                            view = VIEWS[j]
                            idx = i*num_views + j
                            normal = normals_pred[idx]
                            color = images_pred[idx]

                            normal_filename = f"normals_000_{view}.png"
                            rgb_filename = f"rgb_000_{view}.png"
                            normal = save_image(normal, os.path.join(normal_dir, normal_filename))
                            color = save_image(color, os.path.join(scene_dir, rgb_filename))

                            rm_normal = remove(normal)
                            rm_color = remove(color)

                            save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
                            save_image_numpy(rm_color, os.path.join(masked_colors_dir, rgb_filename))

        torch.cuda.empty_cache()
