"""
Dataset related code for training
"""

import mrcfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import iter_mean_std, rescale, clip, fft_to_ps

class MicDataset(Dataset):

    def __init__(self, label_dict, datadir, label_names, transform=None):
        self.label_dict = label_dict
        self.mic_list = list(self.label_dict.keys())
        self.datadir = datadir
        self.transform = transform
        self.label_names = label_names

    def __len__(self):
        return len(self.label_dict)
    
    def __getitem__(self, idx):
        mic_name = self.mic_list[idx]
        mic_path = self.datadir / mic_name
        with mrcfile.open(mic_path) as mrc:
            mic_data = mrc.data
        
        mic_data = torch.tensor(mic_data, dtype=torch.float32)
        
        mic_label_str = self.label_dict[mic_name]
        mic_label = torch.tensor([self.label_names[label_idx].index(name) for label_idx, name in enumerate(mic_label_str)],
                                 dtype=torch.int)
        
        sample = {'data': mic_data, 'label': mic_label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MicRandomCrop(object):
    
    def __init__(self, min_crop_percentage):
        assert min_crop_percentage > 0 and min_crop_percentage <= 1, f'Minimum crop percentage ({min_crop_percentage}) is outside of the range of 0 - 1.'
        self.min_crop_percentage = min_crop_percentage

    def __call__(self, sample):
        mic_data = sample['data']

        h, w = mic_data.shape
        rand_percentage = torch.rand(1).item()*(1-self.min_crop_percentage) + self.min_crop_percentage
        new_size = round(rand_percentage * min(h,w))//2*2

        rand_crop_new_size = transforms.RandomCrop((new_size,new_size))

        return {'data': rand_crop_new_size(mic_data), 'label': sample['label']}

class MicRandomFlip(object):
    
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5):
        assert hflip_prob >= 0 and hflip_prob <= 1, f'Horizontal flip probability ({hflip_prob}) is outside of the range of 0 - 1.'
        assert vflip_prob >= 0 and vflip_prob <= 1, f'Vertical flip probability ({vflip_prob}) is outside of the range of 0 - 1.'
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __call__(self, sample):
        mic_data = sample['data']

        rand_hflip = transforms.RandomHorizontalFlip(self.hflip_prob)
        rand_vflip = transforms.RandomVerticalFlip(self.vflip_prob)
        
        return {'data': rand_vflip(rand_hflip(mic_data)), 'label': sample['label']}

class MicAppendPS(object):
    
    def __init__(self):
        pass

    def __call__(self, sample):
        mic_data = sample['data']
        mic_size = mic_data.shape[0]
        mic_data_fft = torch.fft.rfft2(mic_data)
        
        mic_data_ps = fft_to_ps(mic_data_fft)

        mic_data_with_ps = torch.cat((
            torch.reshape(mic_data, (1, mic_size, mic_size)),
            torch.reshape(mic_data_ps, (1, mic_size, mic_size))),0)
        
        return {'data': mic_data_with_ps, 'label': sample['label']}

class MicReshape(object):
    
    def __init__(self):
        pass

    def __call__(self, sample):
        mic_data = sample['data']
        mic_size = mic_data.shape[0]
        mic_data_reshaped = torch.reshape(mic_data, (1, mic_size, mic_size))
        
        return {'data': mic_data_reshaped, 'label': sample['label']}

class MicResize(object):
    
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        mic_data = sample['data']

        resize_output_size = transforms.Resize(self.output_size,antialias=True)
        
        return {'data': resize_output_size(mic_data), 'label': sample['label']}

class MicNormalize(object):
    
    def __init__(self,new_mean=0,new_std=1,num_std_to_clip=2.5):
        self.new_mean = new_mean
        self.new_std = new_std
        self.num_std_to_clip = num_std_to_clip

    def __call__(self, sample):
        mic_data = sample['data']

        mic_data_scaled_clipped = torch.zeros(*mic_data.shape)
        for idx, channel in enumerate(mic_data):
            mean, std = iter_mean_std(channel)
            channel_scaled = rescale(channel,mean,std,self.new_mean,self.new_std)
            channel_scaled_clipped = clip(channel_scaled,self.new_mean,self.new_std,self.num_std_to_clip)
            mic_data_scaled_clipped[idx] = channel_scaled_clipped
        
        return {'data': mic_data_scaled_clipped, 'label': sample['label']}

def mic_transforms(no_ps=False):
    if no_ps:
        return {'train': transforms.Compose([MicRandomCrop(0.8),
                                             MicRandomFlip(),
                                             MicReshape(),
                                             MicResize(384),
                                             MicNormalize(0,1)]),
                'val': transforms.Compose([MicReshape(),
                                           MicResize(384),
                                           MicNormalize(0,1)])
               }
    else:
        return {'train':transforms.Compose([MicRandomCrop(0.8),
                                            MicRandomFlip(),
                                            MicAppendPS(),
                                            MicResize(384),
                                            MicNormalize(0,1)]),
                'val':transforms.Compose([MicAppendPS(),
                                          MicResize(384),
                                          MicNormalize(0,1)])
               }