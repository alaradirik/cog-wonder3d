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

import linecache
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=30):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def run_wonder3d(num_steps=3000, seed=None):
    tracemalloc.start()

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

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    return [Path("/tmp/mesh_files.zip"), Path("/tmp/output.mp4")]
