import os
from typing import List
from PIL import Image
from omegaconf import OmegaConf
from rembg import remove
from cog import BasePredictor, Input, Path

from utils.misc import load_config 
from utils.image_utils import preprocess
from utils.file_utils import download_model   
from run_mvdiffusion import MultiViewDiffusion
from instant_nsr.launch import run_wonder3d


CHECKPOINT_URLS = [
    ("https://weights.replicate.delivery/default/wonder3d/wonder3d-unet.tar", "/src/ckpts"),
    ("https://weights.replicate.delivery/default/wonder3d/wonder3d-sd-variations.tar", "/src/sd-image-variations-diffusers")
]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        print("Downloading checkpoints and config...")
        for (CKPT_URL, target_folder) in CHECKPOINT_URLS:
            if not os.path.exists(target_folder):
                download_model(CKPT_URL, target_folder)

        self.mvdiffusion_model = MultiViewDiffusion()

    def predict(
        self,
        image: Path = Input(description="Input image to convert to 3D", default=None),
        num_steps: int = Input(description="Number of iterations", ge=100, le=10000, default=3000),
        remove_bg: bool = Input(
            description="Whether to remove image background. Set to false only if uploading image with removed background.",
            default=True
        ),
        random_seed: int = Input(
            description="Random seed for reproducibility, leave blank to randomize output",
            default=None
        )
    ) -> List[Path]:
        # Load image, optionally remove background 
        preprocess(image_path=str(image), remove_bg=remove_bg)

        # Generate alternate views
        self.mvdiffusion_model.generate_views(seed=random_seed)
        print("generated views")

        # Geometry fusion to generate 3D asset
        filepaths = run_wonder3d(num_steps=num_steps, seed=random_seed)
        return filepaths
