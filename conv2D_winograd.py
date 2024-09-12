import sys
import torch
import numpy as np
from torch import tensor
import torch.nn.functional as F
import pdb
class Winograd(torch.nn.Module):
    B = tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, -1.0, 1.0],
         [-1.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, -1.0]])
    B_T = B.transpose(1, 0)
    G = tensor(
        [[1.0, 0.0, 0.0],
         [0.5, 0.5, 0.5],
         [0.5, -0.5, 0.5],
         [0.0, 0.0, 1.0]])
    G_T = G.transpose(1, 0)
    A = tensor([[1.0, 0.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [0.0, -1.0]])
    A_T = A.transpose(1, 0)

    def __init__(self, kernel, padding=1):
        super(Winograd, self).__init__()
        self.kernel = kernel
        self.padding = padding
        self.U = torch.einsum('im,okml->okil', self.G, torch.einsum('okmj,jl->okml', self.kernel, self.G_T))
        self.out_channels, self.in_channels, self.kernel_h, self.kernel_w = kernel.size()  # K=output channels, Cprime=input channels, r=rprime=3
        assert self.kernel_h == self.kernel_w == 3, "Only 3x3 filters are supported."

    def forward(self, x):
        """
            Compute Winograd convolution with output size 2x2 and filter size 3x3 for arbitrary batch size and channels.
            
            :param input: (N, C, H, W) Input tensor.
            :param filter: (K, C, 3, 3) Filter tensor.
            :return: (N, K, H_out, W_out) Output tensor.
        """

        if self.padding > 0:
            x = torch.nn.functional.pad(input=x, pad=(self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        
        
        batch_size, in_channels, H, W = x.size()  # N=batch size, C=input channels, H=W=input height/width
        
        # assert H == W, "Only square input supported for now."
        assert self.in_channels == in_channels, "Input channels and filter input channels must match."

        m = 2  # Output tile size (2x2)
        a = m + self.kernel_h - 1  # Tile size (4x4)
        overlap = self.kernel_h - 1  # Overlap between tiles
        
        # Calculate the number of tiles
        Tw = (W - a) // overlap + 1  # Number of tiles along each dimension
        Th = (H - a) // overlap + 1  # Number of tiles along each dimension
    
        # Transform filters using Winograd G matrix    
        input_unfold = F.unfold(x, kernel_size=(a, a), stride=overlap)
        input_unfold = input_unfold.view(batch_size, in_channels, a, a, Tw * Th)
        V = torch.einsum('ji,ncilm->ncjlm', Winograd.B_T, torch.einsum('ncijm,jl->ncilm', input_unfold, Winograd.B))
        # Element-wise multiplication of transformed filters and inputs
        M = torch.einsum('kcij,ncijt->nkijt', self.U, V)
        # Apply the inverse Winograd transform
        H_out = H - self.kernel_h + 1  # Output height (and width)
        W_out = W - self.kernel_h + 1  # Output height (and width)
        # Y = torch.zeros(N, K, H_out, W_out, device=input.device)
        Ma = torch.einsum('nkilt,lj->nkijt', M, Winograd.A)
        Mb = torch.einsum('ij,nkjlt->nkilt', Winograd.A_T, Ma)
        Mb = Mb.permute(0, 1, 4, 2, 3)  # (N, K, T, m, m) -> (N, K, Tw*Th, m, m)
        # Reshape Mb to (N, K, H_out, W_out) by merging tiles into spatial dimensions.
        Mb = Mb.view(batch_size, self.out_channels, Th, Tw, m, m)  # Reshape to group tiles by height and width
        Y = Mb.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, self.out_channels, H_out, W_out)  # Final output

        return Y
    
if __name__ == "__main__":
    
    x = torch.randn(size=(3, 2, 12, 8))
    
    y = torch.randn(size=(4, 2, 3, 3))
    pad=1
    expect = torch.nn.functional.conv2d(x, y, padding=pad)
    wino = Winograd(kernel=y, padding=pad)
    result1 = wino(x)
    
    np.testing.assert_array_almost_equal(x=expect, y=result1, decimal=5,
        err_msg="The expected array x and computed y are not almost equal.")
    
    print('PASS')