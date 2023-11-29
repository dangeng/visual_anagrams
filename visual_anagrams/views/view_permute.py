import torch
from einops import rearrange

from .permutations import get_inv_perm
from .view_base import BaseView

class PermuteView(BaseView):
    def __init__(self, perm_64, perm_256):
        '''
        Implements arbitrary pixel permutations, for a given permutation. 
            We need two permutations. One of size 64x64 for stage 1, and 
            one of size 256x256 for stage 2.

        perm_64 (torch.tensor) :
            Tensor of integer indexes, defining a permutation, of size 64*64

        perm_256 (torch.tensor) :
            Tensor of integer indexes, defining a permutation, of size 256*256
        '''

        assert perm_64.shape == torch.Size([64*64]), \
            "`perm_64` must be a permutation tensor of size 64*64"

        assert perm_256.shape == torch.Size([256*256]), \
            "`perm_256` must be a permutation tensor of size 256*256"

        # Get random permutation and inverse permutation for stage 1
        self.perm_64 = perm_64
        self.perm_64_inv = get_inv_perm(self.perm_64)

        # Get random permutation and inverse permutation for stage 2
        self.perm_256 = perm_256
        self.perm_256_inv = get_inv_perm(self.perm_256)

    def view(self, im):
        im_size = im.shape[-1]
        perm = self.perm_64 if im_size == 64 else self.perm_256
        num_patches = im_size

        # Permute every pixel in the image
        patch_size = 1

        # Reshape into patches of size (c, patch_size, patch_size)
        patches = rearrange(im, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=patch_size, 
                            p2=patch_size)

        # Permute
        patches = patches[perm]

        # Reshape back into image
        im_rearr = rearrange(patches, 
                             '(h w) c p1 p2 -> c (h p1) (w p2)', 
                             h=num_patches, 
                             w=num_patches, 
                             p1=patch_size, 
                             p2=patch_size)
        return im_rearr

    def inverse_view(self, noise):
        im_size = noise.shape[-1]
        perm_inv = self.perm_64_inv if im_size == 64 else self.perm_256_inv
        num_patches = im_size

        # Permute every pixel in the image
        patch_size = 1

        # Reshape into patches of size (c, patch_size, patch_size)
        patches = rearrange(noise, 
                            'c (h p1) (w p2) -> (h w) c p1 p2', 
                            p1=patch_size, 
                            p2=patch_size)

        # Apply inverse permutation
        patches = patches[perm_inv]

        # Reshape back into image
        im_rearr = rearrange(patches, 
                             '(h w) c p1 p2 -> c (h p1) (w p2)', 
                             h=num_patches, 
                             w=num_patches, 
                             p1=patch_size, 
                             p2=patch_size)
        return im_rearr

    def make_frame(self, im, t):
        # TODO: Implement this, as just moving pixels around
        raise NotImplementedError()


