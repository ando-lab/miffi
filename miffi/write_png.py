"""
Write png files for micrographs
"""

import logging
import argparse
from pathlib import Path
import torch
import matplotlib
from matplotlib import pyplot as plt
from .datasets import MicDataset, mic_transforms
from .utils import iter_mean_std, rescale, clip, crop_in_fourier, fft_to_ps, get_miclist, get_preprocess_param
from multiprocessing import Pool
from tqdm import tqdm

logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument(
        '-f',
        '--miclist-file',
        type=Path,
        help="Path to file that contains the micrograph list. If file extension is .star, micrograph list will be parsed from the star file. If file extension is .csg, microgrph list will be obtained from the cs file specified in the csg file. If containing plain text encoded by utf-8, micrograph names will be parsed from each line in the file. Else, pickle will be used to load object from the file which is assumed to be a list."
    )
    parser.add_argument(
        '-d',
        '--datapath',
        type=Path,
        default=Path.cwd(),
        help="Path to be prepended to those obtained from micrograph list file"
    )
    parser.add_argument(
        '--micdir',
        type=Path,
        help="Path to directory containing input micrographs. Used for specifying micrographs with file name matching. This parameter will only be used if no micrograph list file is specified."
    )
    parser.add_argument(
        '-w',
        '--wildcard',
        default='*.mrc',
        help="Wildcard for input micrograph file name matching. This parameter will only be used if no micrograph list file is specified."
    )
    parser.add_argument(
        '-o',
        '--outdir',
        type=Path,
        default=Path.cwd(),
        help="Path to directory for outputting png files",
    )
    parser.add_argument(
        '-q',
        '--no-preprocessing',
        action='store_true',
        help="Do not preprocess (split and downsample) before writing png files",
    )
    parser.add_argument(
        '--downsample-pixel-size',
        type=float,
        default=1.5,
        help="Pixel size to downsample the micrograph to in unit of angstrom",
    )
    parser.add_argument(
        '-s',
        '--max-split',
        type=int,
        default=None,
        help="Maximum number of micrographs to split non-square micrographs into (starting from top/left)",
    )
    parser.add_argument(
        '-p',
        '--pixel-size',
        type=float,
        default=None,
        help="Pixel size of input micrographs in the unit of angstrom (overwrites pixel size obtained from micrograph mrc files)",
    )
    parser.add_argument(
        '-r',
        '--resize-size',
        type=int,
        default=None,
        help="Size to use for resizing micrographs before normalization, only applicable if preprocessing. Default behvaiour is no additional resizing after Fourier crop.",
    )
    parser.add_argument(
        '--vmin',
        type=float,
        default=-2.5,
        help="Vmin value to use during plotting",
    )
    parser.add_argument(
        '--vmax',
        type=float,
        default=2.5,
        help="Vmax value to use during plotting",
    )
    parser.add_argument(
        '--dpi',
        type=float,
        default=150,
        help="Dpi value to use during plotting",
    )
    parser.add_argument(
        '-n',
        '--num-process',
        type=int,
        default=8,
        help="Number of parallel subprocesses to use for writing png files",
    )

def plot_and_save(mic_path,outdir,data_idx,mic_data,ps_data,vmin=-2.5,vmax=2.5,dpi=150):
    ori_mic_name = mic_path.name
    png_name = ori_mic_name.split('.mrc')[0]+f'_{data_idx}.png'
    path_to_save_file = outdir / png_name
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mic_data,cmap='gray',vmin=vmin,vmax=vmax)
    ax[0].set_axis_off()
    ax[1].imshow(ps_data,cmap='gray',vmin=vmin,vmax=vmax)
    ax[1].set_axis_off()
    plt.savefig(path_to_save_file, format='png', bbox_inches='tight',dpi=dpi)
    plt.close(fig)

def write_png(dataset, idx, mic_path, path_to_outdir, vmin, vmax, dpi):
    mic_data = dataset[idx]
    if isinstance(mic_data, list): # preprocessing
        for data_idx, datai in enumerate(mic_data):
            plot_and_save(mic_path, path_to_outdir, data_idx, datai[0,:], datai[1,:], vmin, vmax, dpi)
    else: # no preprocessing
        mic_data_fft = torch.fft.rfft2(mic_data)
        mic_data_ps = fft_to_ps(mic_data_fft)
        mic_data_ps_fft = torch.fft.rfft2(mic_data_ps)
        small_dim = min(mic_data.shape)
        mic_data_ps_fft_crop = crop_in_fourier(mic_data_ps_fft, small_dim, small_dim)
        mic_data_ps_sq = torch.fft.irfft2(mic_data_ps_fft_crop)

        mic_data_all = [mic_data, mic_data_ps_sq]
        mic_data_all_norm = []
        for channel in mic_data_all:
            mean, std = iter_mean_std(channel)
            channel_scaled = rescale(channel,mean,std,0,1)
            channel_scaled_clipped = clip(channel_scaled,0,1,2.5)
            mic_data_all_norm.append(channel_scaled_clipped)
        
        plot_and_save(mic_path, path_to_outdir, 0, mic_data_all_norm[0], mic_data_all_norm[1], vmin, vmax, dpi)

def write_png_star(input_list):
    return write_png(*input_list)

def main(args):
    mic_list = get_miclist(args.miclist_file, args.micdir, args.wildcard)
    
    if args.no_preprocessing:
        dataset = MicDataset(mic_list, args.datapath, None, None)
        logger.info(f"Write png files without preprocessing")
    else:
        transpose_mic, resized_small_dim, num_split = get_preprocess_param(args.datapath / mic_list[0], args.pixel_size, 
                                                                           args.downsample_pixel_size, args.max_split, True)
        
        resize_size = resized_small_dim
        if args.resize_size is not None:
            resize_size = min(resize_size, args.resize_size)
            logger.info(f"Will resize micrographs and power spectrum to {resize_size} pixels before normalization")
        
        dataset = MicDataset(mic_list, args.datapath, None,
                             transform=mic_transforms(resized_small_dim, num_split, transpose_mic, False, 
                                                      resize_size=resize_size))
    
    if (not args.outdir.exists()): args.outdir.mkdir()
    
    write_png_inputs = [[dataset, idx, mic_path, args.outdir, args.vmin, args.vmax, args.dpi] for idx, mic_path in enumerate(mic_list)]
    
    logger.info(f"Start writing png files with {args.num_process} processes")
    with Pool(processes=args.num_process) as pool:
        result = list(tqdm(pool.imap(write_png_star, write_png_inputs), total=len(write_png_inputs), unit='mic', smoothing=0))
    
    logger.info("All done")

if __name__ == '__main__':
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())