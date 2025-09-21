#learn h_direction. Iterate through many images.
import argparse
import logging
import yaml
import sys
import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint  # Added for gradient checkpointing
import torchvision.utils as tvu
import cv2
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from models.ddpm.diffusion import DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from facexformer.inference import get_attribute
from facexformer.network import FaceXFormer
from face_embedder import FaceEmbeddingModel

class FLUID(object):
    def __init__(self, args, config, device=None):
        # Initialize args, config, and device
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # Calculate beta schedule and related variables
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,  # 0.0001
            beta_end=config.diffusion.beta_end,  # 0.02
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps  # 1000
        )

        # Convert beta schedule to tensor
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        # Calculate alphas and cumulative product
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        # Calculate log variance: fixedlarge - use large fixed variance / fixedsmall - use small fixed variance
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))
        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def invert(self, x0, model, return_latent=False):
        """Invert images into the latent space of the diffusion model."""
        # Prepare for latent pair generation
        print("Preparing identity latent")
        # Ensure exp_id is defined
        if getattr(self.args, "exp", None) is not None:
            exp_id = os.path.split(self.args.exp)[-1]
        else:
            exp_id = "default_exp"  # Default value if self.args.exp is not defined
        # seq_inv: sequence for reverse diffusion steps
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        learn_sigma = False
        img_lat_pairs = []

        x = x0.clone()

        with torch.no_grad():
            # Perform inversion
            with tqdm(total=len(seq_inv), desc=f"Inversion process") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)

                    x, mid_h_g = denoising_step(x, t=t, t_next=t_prev, models=model, #mid_h_g == h-space feature
                                    logvars=self.logvar,
                                    sampling_type='ddim',
                                    b=self.betas,
                                    eta=0,
                                    learn_sigma=learn_sigma)
                    if x.dim() == 3:
                        x = x.unsqueeze(0)
                    progress_bar.update(1)
            
            x_lat = x.clone()
            h_lat = mid_h_g.detach().clone()

            if return_latent is True:
                return [x_lat, h_lat]

    def _checkpointed_denoising_step(self, x, t, t_next, model, h_edit=None):
        """Wrapper function for checkpointed denoising step"""
        def custom_forward(x_input, t_input, t_next_input, h_edit_input):
            if h_edit_input is not None:
                return denoising_step(x_input, t=t_input, t_next=t_next_input, models=model,
                                    logvars=self.logvar,
                                    sampling_type=self.args.sample_type,
                                    b=self.betas,
                                    learn_sigma=False,
                                    edit_h=h_edit_input)
            else:
                return denoising_step(x_input, t=t_input, t_next=t_next_input, models=model,
                                    logvars=self.logvar,
                                    sampling_type=self.args.sample_type,
                                    b=self.betas,
                                    learn_sigma=False)
        
        return custom_forward(x, t, t_next, h_edit)
        
    def denoise(self, x, model, h_edit=None, return_latent=False, enable_grad=False, eta=1.0):
        n=1
        learn_sigma = False
        
        # seq_inv: sequence for reverse diffusion steps
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        if enable_grad:
            context_manager = torch.enable_grad
        else:
            context_manager = torch.no_grad

        with context_manager():
            # Perform reverse diffusion (DDIM)
            with tqdm(total=len(seq_inv), desc=f"Generative process") as progress_bar:
                for it, (i, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    # Determine if this step should use checkpointing
                    use_checkpoint = enable_grad and (it % self.args.checkpoint_interval == 0) and self.args.use_checkpoint

                    if t[0]>=400:
                        if use_checkpoint:
                            # Use gradient checkpointing
                            x, h_edit = checkpoint.checkpoint(
                                self._checkpointed_denoising_step,
                                x, t, t_next, model, h_edit,
                                use_reentrant=False
                            )
                        else:
                            x, h_edit = denoising_step(x, t=t, t_next=t_next, models=model,
                                            logvars=self.logvar,
                                            sampling_type=self.args.sample_type,
                                            b=self.betas,
                                            learn_sigma=learn_sigma,
                                            edit_h=h_edit)
                    elif t[0]>200:
                        if use_checkpoint:
                            x, _ = checkpoint.checkpoint(
                                self._checkpointed_denoising_step,
                                x, t, t_next, model, None,
                                use_reentrant=False
                            )
                        else:
                            x, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                            logvars=self.logvar,
                                            sampling_type=self.args.sample_type,
                                            b=self.betas,
                                            learn_sigma=learn_sigma)
                    else:
                        if use_checkpoint:
                            def final_step_forward(x_input, t_input, t_next_input):
                                return denoising_step(x_input, t=t_input, t_next=t_next_input, models=model,
                                                    logvars=self.logvar,
                                                    sampling_type=self.args.sample_type,
                                                    b=self.betas,
                                                    learn_sigma=learn_sigma,
                                                    eta=eta)
                            
                            x, _ = checkpoint.checkpoint(
                                final_step_forward,
                                x, t, t_next,
                                use_reentrant=False
                            )
                        else:
                            x, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                            logvars=self.logvar,
                                            sampling_type=self.args.sample_type,
                                            b=self.betas,
                                            learn_sigma=learn_sigma,
                                            eta=eta)
                    
                    if x.dim() == 3:
                        x = x.unsqueeze(0)
                    progress_bar.update(1)
        timestamp = time.time()  # Get the current timestamp
        readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        if return_latent:
            return x
        else:
            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.exp, f'{readable_time}.png'))
            return None
        
    def process_image(self, img_path):
        n=1
        x0 = Image.open(img_path).convert("RGB")
        x0 = x0.resize((self.config.data.image_size, self.config.data.image_size), Image.Resampling.LANCZOS)
        x0 = np.array(x0)/255
        x0 = torch.from_numpy(x0).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        x0 = x0.to(self.config.device)
        x0 = (x0 - 0.5) * 2.0  # normalize to [-1, 1]
        if x0.dim() == 3:
            x0 = x0.unsqueeze(0)
            
        return x0
    

def parse_args_and_config():
    """Parse command line arguments and config files."""
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    
    # Default arguments
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--mask_config', type=str, required=False, help='Path to the config file for face parser')
    parser.add_argument('--seed', type=int, default=1006, help='Random seed')
    parser.add_argument('--exp', type=str, default='./output', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--ni', type=int, default=1,  help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--align_face', type=int, default=1, help='align face or not')

    # Sampling arguments
    parser.add_argument('--t', type=int, default=600, help='Return step in [0, 1000)') #the goal inversion step
    #parser.add_argument('--t_edit', type=int, default=500, help='In which timestep to perform explicit edit on h-feature')
    parser.add_argument('--n_inv_step', type=int, default=16, help='# of steps during generative process for inversion')
    parser.add_argument('--n_train_step', type=int, default=50, help='# of steps during generative process for train')
    parser.add_argument('--n_test_step', type=int, default=40, help='# of steps during generative process for test')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for optimizing h_direction')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Controls of variance of the generative process')
    parser.add_argument('--start_distance', type=float, default=-150.0, help='Starting distance of the editing space')
    parser.add_argument('--end_distance', type=float, default=1000.0, help='Ending distance of the editing space. If single edit, then use only ending distance')
    parser.add_argument('--edit_img_number', type=int, default=20, help='Number of editing images')

    # Train & Test arguments
    parser.add_argument('--save_train_image', type=int, default=1, help='Whether to save training results during CLIP finetuning')
    parser.add_argument('--n_precomp_img', type=int, default=100, help='# of images to precompute latents')
    parser.add_argument('--n_train_img', type=int, default=50, help='# of training images')
    parser.add_argument('--n_test_img', type=int, default=10, help='# of test images')
    parser.add_argument('--model_path', type=str, default=None, help='Test model path')
    parser.add_argument('--img_path', type=str, default=None, help='Image path for source')
    #parser.add_argument('--tgt_path', type=str, default=None, help='Image path for target')
    parser.add_argument('--deterministic_inv', type=int, default=1, help='Whether to use deterministic inversion during inference')
    parser.add_argument('--hybrid_noise', type=int, default=0, help='Whether to change multiple attributes by mixing multiple models')
    parser.add_argument('--model_ratio', type=float, default=1, help='Degree of change, noise ratio from original and finetuned model.')
    #parser.add_argument('--norm', type=float, default=25.6, help='The maximum norm for h_direction. Helps analyzing the latent cluster of each identity.')
    
    # Added functionality flags
    parser.add_argument('--radius', action='store_true', help='Calculate the radius of the latent space')
    parser.add_argument('--invert', action='store_true', help='Perform image inversion')
    parser.add_argument('--linear_edit', action='store_true', help='Perform linear edit.')
    #parser.add_argument('--slerp_edit', action='store_true', help='Perform slerp edit.')
    parser.add_argument('--facenet', action='store_true', help='FR model for optimization. FaceNet or MagFace.')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use gradient checkpointing to save memory during training')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Interval for gradient checkpointing during training')
    parser.add_argument('--facenet_threshold', type=float, default=1.3, help='Threshold for identity loss')
    parser.add_argument('--magface_threshold', type=float, default=0.7, help='Threshold for identity loss with MagFace')
    parser.add_argument('--h_direction_norm', type=float, default='0.005', help='Starting norm of h_direction.')

    args = parser.parse_args()

    # Parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # Set up device and logging info
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: {}".format(device))
    new_config.device = device

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    """Convert a dictionary to a namespace."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    """Main function to execute the program."""    
    args, config = parse_args_and_config()
    print(f"Config = {config}")
    print(f"Output path= {args.exp}")
    
    os.makedirs(os.path.join(args.exp), exist_ok=True)
    os.makedirs(os.path.join(args.exp,'best'), exist_ok=True)
    os.makedirs(os.path.join('facex_mask',os.path.basename(args.img_path)), exist_ok=True)

    #load facexformer model
    facex = FaceXFormer().to(config.device)
    weights_path = 'facexformer/ckpts/model.pt'
    facex_checkpoint = torch.load(weights_path, map_location=config.device)
    facex.load_state_dict(facex_checkpoint['state_dict_backbone'])
    facex.eval()
    
    if args.facenet:
        face_model = InceptionResnetV1(pretrained='vggface2').eval().to(config.device)
        face_preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        for param in face_model.parameters():
            param.requires_grad = False
    else:
        magface=FaceEmbeddingModel(model_path='MagFace/ckpts/magface_epoch_00025.pth', device=config.device)
    
    # Load the model
    model = DDPM(config)

    if args.model_path: #in case of bringing a new-trained model 
        init_ckpt = torch.load(args.model_path, map_location=config.device)
        if isinstance(init_ckpt, list) and len(init_ckpt)>0:
            init_ckpt=init_ckpt[0]
            
            # Remove the 'module.' prefix from keys if they exist
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in init_ckpt.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            init_ckpt = new_state_dict
        else:
            raise TypeError(f"unexpected checkpoint type.")
    else:
        local_checkpoint = 'checkpoint/celeba_hq.ckpt'
        if os.path.exists(local_checkpoint):
            init_ckpt = torch.load(local_checkpoint, map_location=config.device)
        else:
            raise FileNotFoundError(f"Checkpoint not found at {local_checkpoint}. Please download it manually.")
    print("Model checkpoint loaded successfully.")

    model.load_state_dict(init_ckpt)
    model.to(config.device)
    model = torch.nn.DataParallel(model)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    for param in facex.parameters():
        param.requires_grad = False
    
    # Initialize
    fluid = FLUID(args, config)
    
    face_list=os.listdir(args.img_path)

    for face in face_list:
        img_path=os.path.join(args.img_path,face)
        basename=os.path.basename(img_path)
        name,_=os.path.splitext(basename)
        if os.path.exists(os.path.join(args.exp, name+ '.png')):
            print(f"Image {face} already processed. Skipping...")
            continue
        #src, face_mask_x0=fluid.process_image(img_path=img_path)
        attribute,face_mask_x0 = get_attribute(image=Image.open(img_path), model=facex, device=config.device)
        face_mask_x0 = transforms.functional.resize(
            face_mask_x0.unsqueeze(0),  # Add channel dimension if needed
            size=(256, 256),  # Match your image size
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        
        #for additional experiments: set attribute values
        #attribute[0,21]=0.9 #google 'celeba attributes' for attribute index
        #attribute[0,31]=0.9
        
        # Create inverse mask (high values in non-face regions)
        inverse_mask = 1.0 - face_mask_x0
        # Ensure proper dimensions for broadcasting
        if inverse_mask.dim() < 4:
            inverse_mask = inverse_mask.unsqueeze(0).unsqueeze(0)
        src=fluid.process_image(img_path=img_path)

        tvu.save_image(face_mask_x0, os.path.join('facex_mask',os.path.basename(args.img_path),f'mask_{name}.png')) #face_mask_x0.shape = [3,256,256]

        # Execute the requested function
        if args.radius:
            fluid.radius() #radius inverts and denoises 100 random samples to determine the appropriate timestep to inject h-feature.
        elif args.invert:
            latents=fluid.invert(x0=src,model=model,return_latent=True)
            print(f'x_lat shape: {latents[0].shape}, h_lat shape {latents[1].shape}')
        else:
            latents=fluid.invert(x0=src,model=model,return_latent=True)    
            #latents=(latents[0].detach(), latents[1].detach())
            source_x_flat = latents[0].view(1, -1)
            source_h_flat = latents[1].view(1,-1)

            #initialize h_direction to a very small random value
            h_direction=torch.randn_like(latents[1]).view(1, -1)
            h_direction=h_direction/torch.norm(h_direction)*args.h_direction_norm
            h_direction = torch.nn.Parameter(h_direction)
                
            # Set up optimizer
            optimizer = torch.optim.Adam([h_direction], lr=args.lr)

            # Constants
            strength = args.end_distance  # Using a fixed strength for optimization
            if args.use_checkpoint:
                print("Starting h-feature optimization with gradient checkpointing...")
            best_image = None
            loss_history = []

            #learn the h-feature
            for j in range(args.n_train_step):
                optimizer.zero_grad()
                print(f'h_direction norm: {torch.norm(h_direction.data)}')
                print(f'h_original norm: {torch.norm(source_h_flat)}')
                
                # Clear GPU cache periodically to prevent memory accumulation
                if j % 10 == 0 and args.use_checkpoint:
                    torch.cuda.empty_cache()
                
                # For h-space
                if args.linear_edit: #in case for ablation study: linear vs slerp
                    edited_h_flat = source_h_flat + strength * h_direction
                else:
                    # Compute the norm and unit vector of the source h-feature
                    dynamic_strength = torch.norm(h_direction)
                    h_direction_unit = h_direction / dynamic_strength
                    source_norm = torch.norm(source_h_flat)
                    source_unit = source_h_flat / source_norm

                    # Remove any component of h_direction that is parallel to source_unit
                    proj = (h_direction_unit * source_unit).sum(dim=1, keepdim=True) * source_unit
                    tangent = h_direction_unit - proj
                    tangent = tangent / torch.norm(tangent, dim=1, keepdim=True)
                    
                    angular_strength = dynamic_strength * 1.0
                    
                    # Here, strength is treated as the angular distance in radians.
                    # Move along the great circle defined by source_unit and the tangent direction:
                    edited_h_flat = source_norm * (
                        torch.cos(torch.tensor(angular_strength, device=source_h_flat.device)) * source_unit +
                        torch.sin(torch.tensor(angular_strength, device=source_h_flat.device)) * tangent
                    )
                    print(f"Denoising edited latent with strength {angular_strength:.2f}...")
                edited_h = edited_h_flat.view_as(latents[1])
                
                # Denoise from latent to image (with gradient checkpointing enabled)
                x=latents[0].clone()
                h = edited_h.clone()
                
                denoised=fluid.denoise(x=x, model=model, h_edit=h, return_latent=True, enable_grad=True, eta=1.0)
                #tvu.save_image((denoised + 1) * 0.5, os.path.join(args.exp, name+ '.png'))
                #attribute_denoised = torch.tensor(get_attribute(image=Image.open(os.path.join(args.exp, name+ '.png')), model=facex, device=config.device), dtype=torch.float32)
                
                source_face=((src + 1) * 0.5).clamp(0, 1)
                
                #facenet embedding distance (with gradients)
                result_face = ((denoised + 1) * 0.5).clamp(0, 1)  # Convert to [0,1] range
                if args.facenet:
                    source_embedding = face_model(face_preprocess(source_face))
                    result_embedding = face_model(face_preprocess(result_face))
                    face_distance = torch.norm(source_embedding - result_embedding, p=2, dim=1)
                else:
                    #magface embedding distance
                    source_magface = magface.get_embedding(src)
                    result_magface, _ = magface.get_embedding(denoised, return_quality=True)
                    face_distance = max(F.cosine_similarity(source_magface, result_magface, dim=1).mean(),0.0)
                
                # Calculate attribute KL divergence loss
                attribute_denoised, mask_denoised = get_attribute(image=result_face, model=facex, device=config.device)
                mask_denoised = transforms.functional.resize(
                    mask_denoised.unsqueeze(0),  # Add channel dimension if needed
                    size=(256, 256),  # Match your image size
                    interpolation=transforms.InterpolationMode.BICUBIC
                )
                
                eps=1e-8
                # 1. Clamp and renormalize
                p = attribute.clamp(min=eps)
                p = p / p.sum(dim=1, keepdim=True)

                q = attribute_denoised.clamp(min=eps)
                q = q / q.sum(dim=1, keepdim=True)

                # 2. Compute KL(P‖Q)
                att_diff = F.kl_div(q.log(), p, reduction='batchmean')
                mask_diff = F.mse_loss(mask_denoised, face_mask_x0)
                
                #background loss
                reg = torch.sum(((denoised-src)**2)*inverse_mask)/(256*256)
                
                # Combined loss with identity loss
                if args.facenet:
                    identity_loss = torch.exp(face_distance * -1.0) #exp(-face_distance)
                    loss = identity_loss+reg*0.5 +att_diff + mask_diff #with facenet
                else:
                    identity_loss =face_distance
                    loss = att_diff * 0.5 + identity_loss + reg * 1.0 + mask_diff #with magface
                
                # Combined loss but altering attributes
                #loss = torch.exp(att_diff * -0.1) + face_diff * 0.1 + non_face_diff * 1.0  #+ torch.exp(face_distance*-1.0) #+ reg
                loss_history.append(loss.item())
                
                if len(loss_history)==0 or min(loss_history)==loss:
                    best_image=denoised.detach().clone()
                    best_vector=h_direction.detach().clone()
                
                loss_value=loss.item()
                loss.backward()
                optimizer.step()

                if args.facenet:
                    print(f"Loss: {loss.item():.4f}, FaceNet: {face_distance.item():.4f}, Reg: {reg:.4f}, Att: {att_diff:.4f}")#, Mask: {mask_diff:.4f}") #with facenet
                else:
                    print(f"Loss: {loss.item():.4f}, FaceNet: {face_distance.item():.4f}, Reg: {reg:.4f}, Att: {att_diff:.4f}") #with magface
            
            #save the best image
            #tvu.save_image((best_image.detach().clone() + 1) * 0.5, os.path.join(args.exp, 'best',name+ '.png'))
            
            #save the final step's image
            tvu.save_image((denoised.detach().clone() + 1) * 0.5, os.path.join(args.exp, name+ '.png'))
            print("Final image saved")
            
            #save h_direction vector
            #torch.save(h_direction, os.path.join(args.exp, name + '_h_direction.pt'))
    print("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
