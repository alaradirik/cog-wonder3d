import os
import random
import shutil
import subprocess

from cog import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from instant_nsr.utils.misc import load_config 
import instant_nsr.datasets as datasets
import instant_nsr.systems as systems


def run_wonder3d(num_steps=3000, seed=None):
    exp_dir = './outputs'

    config = load_config("configs/neuralangelo-ortho-wmask.yaml")
    config.trial_name = ''
    config.exp_dir = config.get('exp_dir') or os.path.join(exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(exp_dir, config.trial_name, 'ckpt')
    config.trainer.max_steps = num_steps

    if seed is None:
        seed = random.randint(0, 10000)

    pl.seed_everything(seed)
    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None)

    callbacks = [ModelCheckpoint(dirpath=config.ckpt_dir, **config.checkpoint)]
    
    trainer = Trainer(
        devices=1,
        accelerator='gpu',
        callbacks=callbacks,
        strategy='ddp_find_unused_parameters_false',
        **config.trainer
    )

    trainer.fit(system, datamodule=dm)
    trainer.test(system, datamodule=dm)

    video_path = [f for f in os.listdir(config.save_dir) if f.endswith(".mp4")][0]
    video_path = os.path.join(config.save_dir, video_path)

    shutil.copy(video_path, "/tmp/output.mp4")
    shutil.rmtree("/src/outputs/image")
    shutil.rmtree("/src/outputs/save")
    if os.path.isdir(config.ckpt_dir):
        shutil.rmtree(config.ckpt_dir)

    subprocess.run(["zip", "-r", "/tmp/mesh_files.zip", "outputs"], check=True)
    shutil.rmtree("/src/outputs/")
    shutil.rmtree("/src/lightning_logs")
    os.remove("/src/image.png")

    return [Path("/tmp/mesh_files.zip"), Path("/tmp/output.mp4")]
