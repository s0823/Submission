#!/usr/bin/env python
import os
import sys
import torch
import torch.nn.functional as F

class FaceEmbeddingModel:
    def __init__(self, model_path='magface_epoch_00025.pth', device=None):
        """
        Initialize the face embedding model.
        
        Args:
            model_path (str): Path to the MagFace model checkpoint
            device (torch.device): Device to run the model on. If None, will use CUDA if available
        """
        # Add MagFace paths
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MagFace'))
        
        from MagFace.inference.network_inf import builder_inf
        
        # Set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Create a simple args object for the model builder
        class Args:
            def __init__(self_args):
                self_args.arch = 'iresnet100'
                self_args.embedding_size = 512
                self_args.resume = model_path
                self_args.cpu_mode = False if 'cuda' in str(self.device) else True
                self_args.dist = 1  # Use this if model is trained with dist
        
        self.args = Args()
        
        # Build the model
        self.model = builder_inf(self.args)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        
        # Store original training state
        self.original_training = self.model.training
        
    def get_embedding(self, image_tensor, return_quality=False):
        """
        Get face embedding from a preprocessed image tensor.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor in [-1, 1] range with shape [B, 3, H, W]
            return_quality (bool): If True, returns both the embedding and its magnitude before normalization
                
        Returns:
            torch.Tensor: Face embedding tensor with shape [B, 512]
            torch.Tensor (optional): Magnitude of embedding before normalization (quality score)
        """
        # Make sure input is on the right device
        image_tensor = image_tensor.to(self.device)
        
        # The image_tensor comes in the range [-1, 1], but the model expects [0, 1] normalization
        normalized_tensor = (image_tensor + 1.0) / 2.0
        
        # Resize to 112x112, which is what MagFace expects
        if normalized_tensor.shape[2] != 112 or normalized_tensor.shape[3] != 112:
            normalized_tensor = F.interpolate(normalized_tensor, size=(112, 112), mode='bilinear', align_corners=False)
        
        # Forward pass with gradients enabled
        with torch.set_grad_enabled(True):
            # Store original training state
            was_training = self.model.training
            
            # Temporarily set to eval mode to avoid batch norm issues
            self.model.eval()
            
            try:
                # Get the raw embedding
                raw_embedding = self.model(normalized_tensor)
                
                # Calculate magnitude (quality score)
                quality = torch.norm(raw_embedding, p=2, dim=1)
                
                # Normalize the embedding for face recognition
                embedding = F.normalize(raw_embedding, p=2, dim=1)
            finally:
                # Restore original training state
                if was_training:
                    self.model.train()
        
        if return_quality:
            return embedding, quality
        else:
            return embedding