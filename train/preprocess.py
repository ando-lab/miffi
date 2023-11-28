"""
Preprocess micrographs for training
"""

import logging
import math
import argparse
from pathlib import Path
import mrcfile
import torch
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Pool
from utils import rescale, iter_mean_std, clip, crop_in_fourier, fft_to_ps
from tqdm import tqdm

logging.basicConfig(format='(%(levelname)s|%(filename)s|%(asctime)s) %(message)s', level=logging.INFO, 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument(
        'micdir',
        type=Path,
        help="Path to directory containing input micrographs"
    )
    parser.add_argument(
        '-w',
        '--wildcard',
        default='*.mrc',
        help="Wildcard for input micrograph file searching"
    )
    parser.add_argument(
        '-o',
        '--outdir',
        type=Path,
        default=Path.cwd(),
        help="Path to directory for outputting preprocessed micrographs",
    )
    parser.add_argument(
        '--skip-mrc',
        action='store_true',
        help="Skip writing preprocessed mrc files",
    )
    parser.add_argument(
        '--skip-png',
        action='store_true',
        help="Skip writing png files",
    )
    parser.add_argument(
        '--downsample-pixel-size',
        type=float,
        default=1.5,
        help="Pixel size to downsample the micrograph to in the unit of angstrom",
    )
    parser.add_argument(
        '-n',
        '--num-process',
        type=int,
        default=8,
        help="Number of parallel subprocesses to use for preprocessing",
    )
    parser.add_argument(
        '--rand-subset',
        type=int,
        default=None,
        help="Number of micrographs to randomly choose as a subset for preprocessing",
    )
    parser.add_argument(
        '-s',
        '--max-split',
        type=int,
        default=None,
        help="Maximum number of micrographs to split non-square micrographs into",
    )
    parser.add_argument(
        '-p',
        '--pixel-size',
        type=float,
        default=None,
        help="Pixel size of input micrographs in the unit of angstrom (overwrites pixel size obtained from micrograph mrc files)",
    )

def save_mrc(mic_path,outdir,data_idx,array,pixel_size):
    ori_mic_name = mic_path.name
    preprocessed_mic_name = ori_mic_name.split('.mrc')[0]+f'_preprocessed_{data_idx}.mrc'
    path_to_save_file = outdir / preprocessed_mic_name
    with mrcfile.new(path_to_save_file,overwrite=True) as mrc:
        mrc.set_data(array.numpy().astype('float16'))
        mrc.voxel_size = pixel_size

def plot_and_save(mic_path,outdir,data_idx,mic_data,ps_data,vmin=-2.5,vmax=2.5,dpi=150):
    ori_mic_name = mic_path.name
    preprocessed_png_name = ori_mic_name.split('.mrc')[0]+f'_preprocessed_{data_idx}.png'
    path_to_save_file = outdir / preprocessed_png_name
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mic_data,cmap='gray',vmin=vmin,vmax=vmax)
    ax[0].set_axis_off()
    ax[1].imshow(ps_data,cmap='gray',vmin=vmin,vmax=vmax)
    ax[1].set_axis_off()
    plt.savefig(path_to_save_file, format='png', bbox_inches='tight',dpi=dpi)
    plt.close(fig)

def preprocess_mic(idx, mic, skip_mrc, skip_png, num_split, mic_size, transpose_mic, resized_small_dim, resized_pixel_size, path_to_mic_outdir, path_to_png_outdir):
    try:
        with mrcfile.open(mic, permissive=True) as mrc:
            data = mrc.data
            data = torch.tensor(data, dtype=torch.float32)
        if transpose_mic:
            data = data.T
        data_split = []
        for idx in range(num_split):
            if idx == math.ceil(mic_size[1]/mic_size[0]) - 1:
                data_split.append(data[:,(mic_size[1]-mic_size[0]):])
            else:
                data_split.append(data[:,idx*mic_size[0]:(idx+1)*mic_size[0]])
        
        data_split_fft = [torch.fft.rfft2(data_i) for data_i in data_split]
        data_split_fft_resized = [crop_in_fourier(data_i,resized_small_dim,resized_small_dim) for data_i in data_split_fft]
        data_split_resized = [torch.fft.irfft2(data_i) for data_i in data_split_fft_resized]
    
        data_mean_std = [iter_mean_std(data_i) for data_i in data_split_resized]
        data_split_resized_scaled = [rescale(data_i,*mean_std) for data_i, mean_std in zip(data_split_resized, data_mean_std)]
    
        if not skip_mrc:
            for data_idx, data_i in enumerate(data_split_resized_scaled):
                save_mrc(mic,path_to_mic_outdir,data_idx,data_i,resized_pixel_size)
        
        if not skip_png:
            data_split_ps = [fft_to_ps(data_i) for data_i in data_split_fft_resized]
            ps_mean_std = [iter_mean_std(data_i) for data_i in data_split_ps]
            data_split_ps_scaled = [rescale(data_i,*mean_std) for data_i, mean_std in zip(data_split_ps, ps_mean_std)]
            data_split_ps_scaled_clipped = [clip(data_i,0,1) for data_i in data_split_ps_scaled]
            
            for data_idx, (data_i, ps_i) in enumerate(zip(data_split_resized_scaled, data_split_ps_scaled_clipped)):
                plot_and_save(mic,path_to_png_outdir,data_idx,data_i,ps_i)

    except Exception as e:
        raise Exception(f"Exception encountered during processing of {mic}") from e

def preprocess_mic_star(input_list):
    return preprocess_mic(*input_list)

def main(args):
    mic_list = list(args.micdir.glob(args.wildcard))
    if args.rand_subset is not None:
        rand_idx = torch.randperm(len(mic_list))[:args.rand_subset]
        mic_list = [mic_list[idx.item()] for idx in rand_idx]
    
    with mrcfile.open(mic_list[0], permissive=True) as mrc:
        data = mrc.data
        ori_pixel_size = mrc.voxel_size['x']
    if args.pixel_size is not None:
        ori_pixel_size = args.pixel_size
    assert ori_pixel_size > 0.01, f"Invalid original micrograph pixel size! (value: {ori_pixel_size:.5f} angstroms)"
    downsample_pixel_size = args.downsample_pixel_size
    if ori_pixel_size > downsample_pixel_size:
        logger.warning(f"Original pixel size ({ori_pixel_size}) larger than downsample pixel size ({downsample_pixel_size})! Proceed without downsampling ...")
        downsample_pixel_size = ori_pixel_size
    logger.info(f"Downsampling micrograph from its original pixel size {ori_pixel_size:.5f} angstrom to {downsample_pixel_size} angstrom")
    
    transpose_mic = False
    mic_size = list(data.shape)
    if mic_size[0] > mic_size[1]:
        logger.info("First micrograph dimension longer than second dimension, micrograph will be transposed")
        mic_size.reverse()
        transpose_mic = True
    resized_small_dim = round(mic_size[0] * ori_pixel_size / downsample_pixel_size) // 2 * 2
    resized_pixel_size = round(mic_size[0] * ori_pixel_size / resized_small_dim, 3)

    num_split = math.ceil(mic_size[1]/mic_size[0])
    if args.max_split is not None:
        num_split = min(num_split,args.max_split)
    
    if mic_size[0] != mic_size[1]: logger.info(f"Non-square micrograph, will divide into {num_split} square micrograph(s) along long dimension starting from top/left")

    path_to_mic_outdir = args.outdir / 'preprocessed'
    path_to_png_outdir = args.outdir / 'preprocessed_png'
    if (not args.skip_mrc) and (not path_to_mic_outdir.exists()): path_to_mic_outdir.mkdir()
    if (not args.skip_png) and (not path_to_png_outdir.exists()): path_to_png_outdir.mkdir()

    preprocess_inputs = [[idx, mic, args.skip_mrc, args.skip_png, num_split, mic_size, transpose_mic, resized_small_dim, resized_pixel_size, path_to_mic_outdir, path_to_png_outdir] for idx, mic in enumerate(mic_list)]

    logger.info(f"Start preprocessing with {args.num_process} processes")
    with Pool(processes=args.num_process) as pool:
        result = list(tqdm(pool.imap(preprocess_mic_star, preprocess_inputs), total=len(preprocess_inputs), unit='mic', smoothing=0))
    
    logger.info("All done")

if __name__ == '__main__':
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())