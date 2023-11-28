"""
Utilities for training
"""

import torch

def rescale(im, m1, s1, m2 = 0, s2 = 1):
    im_new = m2 + (im - m1) * s2 / s1
    return im_new

def iter_mean_std(data, num_iter=3, num_std=3):
    lower_bound = torch.amin(data)
    upper_bound = torch.amax(data)
    for iterations in range(num_iter):
        mean = torch.mean(data[(data>lower_bound)&(data<upper_bound)])
        std = torch.std(data[(data>lower_bound)&(data<upper_bound)])
        lower_bound = mean - num_std*std
        upper_bound = mean + num_std*std
    return mean, std

def clip(im,m,s, num_std=2.5):
    im[im>(m+num_std*s)] = m+num_std*s
    im[im<(m-num_std*s)] = m-num_std*s
    return im

def crop_in_fourier(fft,new_x,new_y):
    return torch.cat((fft[:new_y//2,:(new_x//2+1)],fft[(fft.shape[0]-new_y//2):,:(new_x//2+1)]), dim=0)

def fft_to_ps(fft):
    fft_abs = torch.abs(fft)
    ps_size = [fft_abs.shape[0],(fft_abs.shape[1]-1)*2]
    ps = torch.zeros(ps_size[0],ps_size[1])
    ps[ps_size[0]//2:,ps_size[1]//2:] = fft_abs[:ps_size[0]//2,:ps_size[1]//2]
    ps[:ps_size[0]//2,ps_size[1]//2:] = fft_abs[ps_size[0]//2:,:ps_size[1]//2]
    ps[:,:ps_size[1]//2] = torch.flip(ps[:,ps_size[1]//2:],[0, 1])
    return ps
