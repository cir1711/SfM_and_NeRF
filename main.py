import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time 

L_embed = 6
def posenc(x):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, -1)

embed_fn = posenc

class Model(torch.nn.Module):
  def __init__(self, filter_size=128, num_encoding_functions=6):
    super(Model, self).__init__()
    self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
    self.layer2 = torch.nn.Linear(filter_size, filter_size)
    self.layer3 = torch.nn.Linear(filter_size, 4)
    self.relu = torch.nn.functional.relu
  
  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.relu(self.layer2(x))
    x = self.layer3(x)
    return x

def get_rays(H, W, focal, c2w):
    columns = torch.arange(W, dtype=torch.float32, device='cuda')
    rows = torch.arange(H, dtype=torch.float32, device='cuda')
    i, j = torch.meshgrid(columns, rows, indexing='xy')
    dirs = torch.stack([(i - W * .5) / focal,
                            -(j - H * .5) / focal,
                            -torch.ones_like(i)
                           ], dim=-1)
    ends = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    origins = c2w[:3, -1].expand(ends.shape)
    return origins, ends


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
  dim = -1
  cumprod = torch.cumprod(tensor, dim)
  cumprod = torch.roll(cumprod, 1, dim)
  cumprod[..., 0] = 1.
  return cumprod

def batchify(fn, chunk=1024*32):
        return lambda inputs : torch.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples):
    z_vals = torch.linspace(near, far, steps=N_samples, device='cuda')
    z_vals += torch.rand(z_vals.size(), device='cuda') * (far - near) /N_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :]* z_vals[...,:,None]
    pts_flat = torch.reshape(pts, (-1, 3))

    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    
    unflattened_shape = list(pts.shape[:-1]) + [4]
    raw = torch.reshape(raw, unflattened_shape)

    sigma_a = F.relu(raw[...,3])
    rgb = torch.sigmoid(raw[...,:3])
    
    one_e_10 = torch.tensor([1e10], device='cuda')
    dists = torch.cat((z_vals[..., 1:] - z_vals[..., :-1],
                  one_e_10.expand(z_vals[..., :1].shape)), dim=-1)
 
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * torch.cumprod(1. - alpha + 1e-10, dim=-1)

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map

device = 'cuda'
data = np.load('tiny_nerf_data.npz')
images = data['images'] 
poses = data['poses']
focal = torch.from_numpy(data['focal']).to(device)

H, W = images.shape[1:3]

testimg, testpose = images[101], poses[101]
testimg = torch.from_numpy(testimg).to(device)
testpose = torch.from_numpy(testpose).to(device)
images = torch.from_numpy(images[:100, ..., :3]).to(device)
poses =  torch.from_numpy(poses[:100]).to(device)


model = Model().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=5e-4)

N_samples = 64
N_iters = 10000
psnrs = []
iternums = []
i_plot = 100

t = time.time()
for i in range(N_iters + 1):
    img_i = np.random.randint(images.shape[0])
    target = images[img_i].to(device)
    pose = poses[img_i].to(device)
    rays_o, rays_d = get_rays(H, W, focal, pose)
    optimizer.zero_grad()
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
    loss = torch.mean((rgb - target) ** 2)
    loss.backward()
    optimizer.step()
    if i % i_plot == 0:
        print(i, (time.time() - t) / i_plot, 'secs per iter')
        t = time.time()

        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        loss = torch.mean((rgb - testimg) ** 2)
        psnr = -10. * torch.log10(loss)

        psnrs.append(psnr.item())
        iternums.append(i)

        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(rgb.detach().cpu().numpy())
        plt.title(f'Iteration: {i}')
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title('PSNR')
        plt.savefig(f'novel_views/nerf_{i}.png')
        plt.close()
print('Done')
