# -*- coding: utf-8 -*-
"""
PyTorch port of the ghost imaging reconstruction using a deep neural network
constraint (GIDC) algorithm reported in the paper:
Fei Wang et al. 'Far-field super-resolution ghost imaging with a deep neural
network constraint'. Light Sci Appl 11, 1 (2022).
https://doi.org/10.1038/s41377-021-00680-w
Please cite the paper if you find this code offers any help.

Original TensorFlow 1.x implementation by Fei Wang (WangFei_m@outlook.com).

Inputs:
    A_real: illumination patterns (pixels * pixels * pattern numbers)
    y_real: single pixel measurements (pattern numbers)

Outputs:
    x_out: reconstructed image by GIDC (pixels * pixels)
"""

import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F

from GIDC_model_Unet import GIDCUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
data = loadmat('data.mat')
result_save_path = os.path.join('.', 'results')

# create results save path
os.makedirs(result_save_path, exist_ok=True)

# define optimization parameters
img_W = 64
img_H = 64
SR = 0.1                                       # sampling rate
batch_size = 1
lr0 = 0.05                                     # learning rate
TV_strength = 1e-9                             # regularization parameter of Total Variation
num_patterns = int(np.round(img_W * img_H * SR))   # number of measurement times
Steps = 201                                    # optimization steps

if num_patterns > data['patterns'].shape[-1]:
    raise Exception('Please set a smaller SR')

A_real = data['patterns'][:, :, 0:num_patterns]   # illumination patterns
y_real = data['measurements'][0:num_patterns]     # intensity measurements

# DGI reconstruction
print('DGI reconstruction...')
B_aver = 0
SI_aver = 0
R_aver = 0
RI_aver = 0
count = 0
for i in range(num_patterns):
    pattern = data['patterns'][:, :, i]
    count = count + 1
    B_r = data['measurements'][i]

    SI_aver = (SI_aver * (count - 1) + pattern * B_r) / count
    B_aver = (B_aver * (count - 1) + B_r) / count
    R_aver = (R_aver * (count - 1) + np.sum(pattern)) / count
    RI_aver = (RI_aver * (count - 1) + np.sum(pattern) * pattern) / count
    DGI = SI_aver - B_aver / R_aver * RI_aver
# DGI[DGI<0] = 0
print('Finished')

# preprocessing
DGI = np.reshape(DGI, [img_W, img_H]).astype(np.float32)
y_real = np.reshape(y_real, [num_patterns]).astype(np.float32)
A_real = np.reshape(A_real, [img_W, img_H, num_patterns]).astype(np.float32)

# DGI = DGI.T  # sometimes it gives better results
DGI = (DGI - np.mean(DGI)) / np.std(DGI)
y_real = (y_real - np.mean(y_real)) / np.std(y_real)
A_real = (A_real - np.mean(A_real)) / np.std(A_real)

# move data to tensors
#   inpt:     network input (the DGI estimate)        [batch, 1, W, H]
#   patterns: measurement operator (conv weights)     [num_patterns, 1, W, H]
#   y_t:      measured single-pixel intensities       [batch, num_patterns]
inpt = torch.from_numpy(DGI).reshape(batch_size, 1, img_W, img_H).to(device)
patterns = torch.from_numpy(np.transpose(A_real, (2, 0, 1))).reshape(
    num_patterns, 1, img_W, img_H).to(device)
y_t = torch.from_numpy(y_real).reshape(batch_size, num_patterns).to(device)

# prepare for surveillance (the original used a Fortran-order reshape, which for
# a square image is equivalent to a transpose of the natural orientation)
DGI_temp0 = DGI.T
y_real_temp = y_real

# Build the DNN (the physical model y = Ax is embedded in the network).
# y: measurements (known)  A: physical model (known)  x: object (unknown)
model = GIDCUNet().to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr0,
                             betas=(0.5, 0.9), eps=1e-08)


def total_variation(img):
    """Anisotropic total variation, matching tf.image.total_variation."""
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    return dh + dw


print('GIDC reconstruction...')
for step in range(Steps):
    # exponential learning-rate decay: lr0 * 0.90 ** (step / 100)
    lr_temp = lr0 * 0.90 ** (step / 100.0)
    for group in optimizer.param_groups:
        group['lr'] = lr_temp

    out_x, out_y = model(inpt, patterns)
    loss_y = F.mse_loss(y_t, out_y)
    TV_reg = TV_strength * total_variation(out_x)
    loss = loss_y + TV_reg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print('step:%d----y loss:%f----learning rate:%f----num of patterns:%d'
              % (step, loss_y.item(), lr_temp, num_patterns))

        with torch.no_grad():
            x_out = out_x.detach().cpu().numpy()[0, 0]
            y_out = out_y.detach().cpu().numpy().reshape(num_patterns)
        x_out = x_out.T   # match the original display orientation

        plt.subplot(141)
        plt.imshow(DGI_temp0)
        plt.title('DGI')
        plt.yticks([])

        plt.subplot(142)
        plt.imshow(x_out)
        plt.title('GIDC')
        plt.yticks([])

        ax1 = plt.subplot(143)
        plt.plot(y_out)
        plt.title('pred_y')
        ax1.set_aspect(1.0 / ax1.get_data_ratio(), adjustable='box')
        plt.yticks([])

        ax2 = plt.subplot(144)
        plt.plot(y_real_temp)
        plt.title('real_y')
        ax2.set_aspect(1.0 / ax2.get_data_ratio(), adjustable='box')
        plt.yticks([])

        plt.subplots_adjust(hspace=0.25, wspace=0.25)
        plt.show()

        x_save = x_out - np.min(x_out)
        x_save = x_save * 255 / np.max(np.max(x_save))
        x_save = Image.fromarray(x_save.astype('uint8')).convert('L')
        x_save.save(os.path.join(result_save_path,
                                 'GIDC_%d_%d.bmp' % (num_patterns, step)))

print('Finished!')
