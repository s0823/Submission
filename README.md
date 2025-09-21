To run the code, the following model files should be prepared.

1. Download model.pt from [FaceXformer](https://github.com/Kartik-3004/facexformer) and place it in facexformer/ckpts/model.pt
2. Download magface_epoch_00025.pth, which is the first model in the model zoo of [MagFace](https://github.com/IrvingMeng/MagFace?tab=readme-ov-file) and place it in MagFace/ckpts/magface_epoch_00025.pth
3. Please refer to [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP) or [Asyrp](https://github.com/kwonminki/Asyrp_official?tab=readme-ov-file) to download the diffusion model trained on [CelebA-HQ](https://arxiv.org/abs/1710.10196) , create a 'checkpoint' folder and place the ckpt model file in checkpoint/celeba_hq.ckpt


Please install the environment with the following steps:
```
conda create --name fluid python=3.8
conda activate latent
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
cd fluid
pip install -r requirements.txt
pip install facenet-pytorch
pip install kornia
```


To run anonymization, please run the following command:
```
python run_fluid.py --exp output/folder --img_path input/folder --config celeba.yml --t 600 --n_inv_step 16 --n_train_step 50 --lr 0.001 --facenet --linear_edit --end_distance 1000
```
with the following description:
```
--exp: Output directory.
--img_path: Input directory. All images in this directory will be anonymized automatically.
--config: The config file of the pretrained diffusion model.
--t: The inversion step T, which is the starting timestep of optimization.
--n_inv_step: Number of timesteps for both inversion and generation. Increasing this will result in more memory consumption.
--n_train_step: Number of optimization steps per anonymization.
--lr: Learning rate.
--facenet: Using FaceNet as the auxiliary face recognition model for the identity loss. If this command is absent, MagFace is used.
--linear_edit: Using linear edit for anonymization. If this command is absent, tangent edit is used.
--end_distance: Editing strength (only for linear edit).
--use_checkpoint: Enable gradient checkpointing for much less memory usage.
--model_path: Path to your own pretrained diffusion model.
```

To train your own diffusion model and use for editing, please refer to [DDIM](https://github.com/ermongroup/ddim) and make sure to change the input for `--config` and `--model_path` in the command above.



This implementation heavily relies on [Asyrp](https://github.com/kwonminki/Asyrp_official?tab=readme-ov-file), [Boundary Diffusion](https://github.com/L-YeZhu/BoundaryDiffusion), [FaceXformer](https://github.com/Kartik-3004/facexformer), and [MagFace](https://github.com/IrvingMeng/MagFace?tab=readme-ov-file).
