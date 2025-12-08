import torch
import torch.nn.functional as F


def custom_sinc(x):
    return torch.where(torch.abs(x)<1e-6, torch.ones_like(x), (torch.sin(3.1415*x)/(3.1415*x)).to(x.dtype))


def custom_unfold(x, kernel_size=3, padding=1):
    B, C, H, W = x.shape
    p2d = (padding, padding, padding, padding)
    x_pad = F.pad(x, p2d, "replicate")
    x_list = list()
    for ind_i in range(kernel_size):
        for ind_j in range(kernel_size):
            x_list.append(x_pad[:, :, ind_i:ind_i + H, ind_j:ind_j + W])

    x_unfold = torch.cat(x_list, dim=1)

    return x_unfold

def coords_grid(b, h, w, device):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]
    stacks = [x, y]
    grid = torch.stack(stacks, dim=0)  # [2, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W]

    return grid.to(device)

def bilinear_sampler(img, coords, mode='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    W = torch.tensor(img.shape[-1], dtype=img.dtype, device=img.device)
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    out = F.grid_sample(img, grid, mode=mode, align_corners=False)

    return out


def get_pe(h: int, w: int, pe_dim: int, dtype: str, device: str):
    """ relative positional encoding """
    with torch.no_grad():
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, h-1, h).to(device).to(dtype),
                                        torch.linspace(0, w-1, w).to(device).to(dtype), indexing='ij')
        rel_x_pos = (grid_x.reshape(-1, 1) - grid_x.reshape(1, -1)).long()
        rel_y_pos = (grid_y.reshape(-1, 1) - grid_y.reshape(1, -1)).long()

        L = 2 * w + 1
        sig = 5 / pe_dim
        x_pos = torch.linspace(-3, 3, L).to(device).to(dtype).tanh()
        dim_t = torch.linspace(-1, 1, pe_dim//2).to(device).to(dtype)
        pe_x = custom_sinc((dim_t[None, :] - x_pos[:, None]) / sig)

        pe_x = F.normalize(pe_x, p=2, dim=-1)
        rel_pe_x = pe_x[rel_x_pos + w - 1].reshape(h * w, h * w, pe_dim//2).to(dtype)

        L = 2 * h + 1
        sig = 5 / pe_dim
        y_pos = torch.linspace(-3, 3, L).to(device).to(dtype).tanh()
        dim_t = torch.linspace(-1, 1, pe_dim//2).to(device).to(dtype)
        pe_y = custom_sinc((dim_t[None, :] - y_pos[:, None]) / sig)

        pe_y = F.normalize(pe_y, p=2, dim=-1)
        rel_pe_y = pe_y[rel_y_pos + h - 1].reshape(h * w, h * w, pe_dim//2).to(dtype)

        pe = .5*torch.cat([rel_pe_x, rel_pe_y], dim=2)

    return pe.clone()
