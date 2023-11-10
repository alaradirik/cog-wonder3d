## Cog-Wonder3D
Cog wrapper for Wonder3D: Single Image to 3D using Cross-Domain Diffusion based on the original [repository](https://github.com/xxlong0/Wonder3D), see the [paper](https://arxiv.org/abs/2310.15008), [project page](https://www.xxlong.site/Wonder3D/) and [Replicate demo](https://replicate.com/adirik/wonder3d/) for details.

## Basic Usage
Wonder 3D is an image-to-3D model that uses a multi-view diffusion model as its backbone to generate 3D assets in a few minutes. You need to have Cog and Docker installed to run this model locally. Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of Wonder3D to [Replicate](https://replicate.com).

To build the docker image with cog and run a prediction:
```bash
cog predict -i image=@example_images/sample.jpg -i remove_bg=True -i num_steps=3000
```

To start a server and send requests to your locally deployed API:
```bash
cog run -p 5000 python -m cog.server.http
```

## References
```
@misc{long2023wonder3d,
      title={Wonder3D: Single Image to 3D using Cross-Domain Diffusion}, 
      author={Xiaoxiao Long and Yuan-Chen Guo and Cheng Lin and Yuan Liu and Zhiyang Dou and Lingjie Liu and Yuexin Ma and Song-Hai Zhang and Marc Habermann and Christian Theobalt and Wenping Wang},
      year={2023},
      eprint={2310.15008},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
