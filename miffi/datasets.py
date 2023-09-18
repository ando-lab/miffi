"""
Dataset related code for miffi
"""

import math
import mrcfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .utils import iter_mean_std, rescale, clip, crop_in_fourier, fft_to_ps

class MicDataset(Dataset):

    def __init__(self, mic_list, datapath, label_names, transform=None):
        self.mic_list = mic_list
        self.datapath = datapath
        self.transform = transform
        self.label_names = label_names

    def __len__(self):
        return len(self.mic_list)
    
    def __getitem__(self, idx):
        mic_name = self.mic_list[idx]
        mic_path = self.datapath / mic_name
        with mrcfile.open(mic_path, permissive=True) as mrc:
            mic_data = mrc.data
        
        mic_data = torch.tensor(mic_data, dtype=torch.float32)

        if self.transform:
            mic_data = self.transform(mic_data)

        return mic_data

class MicFourierCrop(object):

    def __init__(self, resized_small_dim, num_split, transpose_mic, no_ps):
        self.resized_small_dim = resized_small_dim
        self.num_split = num_split
        self.transpose_mic = transpose_mic
        self.no_ps = no_ps

    def __call__(self, mic_data):
        if self.transpose_mic:
            mic_data = mic_data.T
        data_split = []
        for idx in range(self.num_split):
            if idx == math.ceil(mic_data.shape[1]/mic_data.shape[0]) - 1:
                data_split.append(mic_data[:,(mic_data.shape[1]-mic_data.shape[0]):])
            else:
                data_split.append(mic_data[:,idx*mic_data.shape[0]:(idx+1)*mic_data.shape[0]])
        
        data_split_fft = [torch.fft.rfft2(data_i) for data_i in data_split]
        data_split_fft_resized = [crop_in_fourier(data_i,self.resized_small_dim,self.resized_small_dim) for data_i in data_split_fft]
        data_split_resized = [torch.fft.irfft2(data_i) for data_i in data_split_fft_resized]

        if self.no_ps:
            return [torch.reshape(data_i, (1, self.resized_small_dim, self.resized_small_dim)) for data_i in data_split_resized]
        else:
            data_split_ps = [fft_to_ps(data_i) for data_i in data_split_fft_resized]
            return [torch.cat(
                (torch.reshape(data_i, (1, self.resized_small_dim, self.resized_small_dim)),
                 torch.reshape(ps_i, (1, self.resized_small_dim, self.resized_small_dim))),0) 
                    for data_i, ps_i in zip(data_split_resized, data_split_ps)]

class MicResize(object):
    
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, mic_data):

        resize_output_size = transforms.Resize(self.output_size,antialias=True)
        
        return [resize_output_size(data_i) for data_i in mic_data]

class MicNormalize(object):
    
    def __init__(self,new_mean=0,new_std=1,num_std_to_clip=2.5):
        self.new_mean = new_mean
        self.new_std = new_std
        self.num_std_to_clip = num_std_to_clip

    def __call__(self, mic_data):
        mic_data_scaled_clipped = [torch.zeros(*mic_data[idx].shape) for idx in range(len(mic_data))]
        for data_idx, data_i in enumerate(mic_data):
            for idx, channel in enumerate(data_i):
                mean, std = iter_mean_std(channel)
                channel_scaled = rescale(channel,mean,std,self.new_mean,self.new_std)
                channel_scaled_clipped = clip(channel_scaled,self.new_mean,self.new_std,self.num_std_to_clip)
                mic_data_scaled_clipped[data_idx][idx] = channel_scaled_clipped
        
        return mic_data_scaled_clipped

def mic_transforms(resized_small_dim, num_split, transpose_mic, no_ps, 
                   resize_size=384, mean=0, std=1, num_std_to_clip=2.5):
    return transforms.Compose([MicFourierCrop(resized_small_dim, num_split, transpose_mic, no_ps),
                               MicResize(resize_size),
                               MicNormalize(mean,std,num_std_to_clip)])