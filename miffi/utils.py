"""
Utilities for miffi
"""

import logging
import math
import numpy as np
import yaml
import torch
from pathlib import Path
import pickle
import mrcfile

logger = logging.getLogger(__name__)

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

def is_plain_text_file(file_path):
    with open(file_path, 'rb') as file:
        try:
            file_contents = file.read()
            file_contents.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False

def load_pkl(file):
    with open(file,'rb') as f:
        return pickle.load(f)

def save_pkl(obj, file):
    with open(file,'wb') as f:
        pickle.dump(obj, f)

def load_yaml(file):
    with open(file,'r') as f:
        return yaml.safe_load(f)

def star_to_miclist(file):
    import starfile
    df = starfile.read(file)
    return df['micrographs']['rlnMicrographName'].tolist()

def csg_to_miclist(file):
    csg = load_yaml(file)
    assert 'micrograph_blob' in csg['results'].keys(), "'micrograph_blob' entry not found in the provided csg file!"
    cs_file_name = file.parent/csg['results']['micrograph_blob']['metafile'].replace('>','')
    
    assert cs_file_name.exists(), f"cs file {cs_file_name.name} specified in the csg file does not exist in the same directory!"
    cs = np.load(cs_file_name)
    assert 'micrograph_blob/path' in cs.dtype.names, f"'micrograph_blob/path' entry was not found in the cs file {cs_file_name.name} specified in the csg file!"
    return list(map(lambda x: x.decode('utf-8'), cs['micrograph_blob/path'].tolist()))

def get_miclist(miclist_file, micdir, wildcard):
    if miclist_file is None:
        if micdir is not None:
            mic_list = sorted(list(micdir.glob(wildcard)))
            assert len(mic_list) > 0, f"No micrograph found with file name matching {micdir/wildcard}"
        else:
            raise ValueError("No micrograph input provided")
    else:
        if micdir is not None:
            logger.warning("Using specified micrograph list file. Micrograph directory input will be ignored.")
        if miclist_file.suffix.lower() == '.star':
            logger.info("Load micrograph list from star file")
            mic_list = star_to_miclist(miclist_file)
        elif miclist_file.suffix.lower() == '.csg':
            logger.info("Load micrograph list from cs file specified in provided csg file")
            mic_list = csg_to_miclist(miclist_file)
        elif is_plain_text_file(miclist_file):
            logger.info("Load micrograph list from plain text file")
            with open(miclist_file,'r') as f:
                mic_list = f.read().splitlines()
                mic_list = [pathi.strip() for pathi in mic_list if pathi.strip() != '']
        else:
            logger.info("Load input file as a pickle file containing a list")
            mic_list = load_pkl(miclist_file)
        mic_list = [Path(pathi) for pathi in mic_list]
        assert len(mic_list) > 0, "No micrograph found within the provided file"
    return mic_list

def get_preprocess_param(micpath, pixel_size, downsample_pixel_size, max_split, no_ps):
    with mrcfile.open(micpath, permissive=True) as mrc:
        data = mrc.data
        ori_pixel_size = mrc.voxel_size['x']
    if pixel_size is not None:
        ori_pixel_size = pixel_size
        logger.info(f"Specified pixel size: {ori_pixel_size:.3f} angstrom")
    else:
        logger.info(f"Obtained pixel size from .mrc file header: {ori_pixel_size:.3f} angstrom")
    assert ori_pixel_size > 0.01, f"Invalid original micrograph pixel size! (value: {ori_pixel_size:.3f} angstrom)"
    if ori_pixel_size > downsample_pixel_size:
        logger.warning(f"Original pixel size ({ori_pixel_size:.3f}) larger than downsample pixel size ({downsample_pixel_size:.3f})! Proceed without downsampling (setting downsample pixel size to {ori_pixel_size:.3f} angstrom.)")
        downsample_pixel_size = ori_pixel_size
    if downsample_pixel_size != 1.5 and not no_ps:
        logger.warning(f"Downsample pixel size ({downsample_pixel_size:.3f}) is not 1.5 angstrom. If original miffi models with ps are used, result may be incorrect as models were trained with a downsample pixel size of 1.5 angstrom.")
    logger.info(f"Downsampling micrograph from its original pixel size {ori_pixel_size:.3f} angstrom to {downsample_pixel_size:.3f} angstrom during preprocessing")

    transpose_mic = False
    mic_size = list(data.shape)
    if mic_size[0] > mic_size[1]:
        logger.info("First micrograph dimension longer than second dimension, micrograph will be transposed during preprocessing")
        mic_size.reverse()
        transpose_mic = True
    resized_small_dim = round(mic_size[0] * ori_pixel_size / downsample_pixel_size) // 2 * 2

    num_split = math.ceil(mic_size[1]/mic_size[0])
    if max_split is not None:
        num_split = min(num_split,max_split)
    
    if mic_size[0] != mic_size[1]: logger.info(f"Non-square micrograph, will divide into {num_split} square micrograph(s) along long dimension starting from top/left during preprocessing")

    return transpose_mic, resized_small_dim, num_split

def find_max_catidx(pred,cat_preds,cat_confs):
    index_of_this = torch.nonzero(cat_preds==pred, as_tuple=True)[0]
    max_out_idx = torch.argmax(cat_confs[index_of_this][:,pred])
    return index_of_this[max_out_idx]

def parse_combine_rule(rule):
    rule_dict = {}
    rule_cat_all = rule.split(':')
    for rule_cat in rule_cat_all:
        rule_label_all = rule_cat.split(',')
        rule_label_all_list = [[int(rule_label[idx]) for idx in range(3)] for rule_label in rule_label_all[1:]]
        rule_dict[int(rule_label_all[0])] = rule_label_all_list
    return rule_dict
    
def combine_preds(mic_preds, mic_pred_confidence, label_names, rule = None):
    '''
    In the returned tensor, first dimension contains pred, second dimension contains the index corresponding to the original split with highest confidence.
    If predictions are different for different splits, they will be combined according to rule given.
    Rule definition: catagory_number, contains_label_number set_to_label_number find_max_confidence_for_label_number, next_label... : next_catagory...
    Conditions are evaluated squentially. For catagory not mentioned, binary catagorization is assumed with the default rule 000
    '''
    combined_preds = torch.zeros(2,*mic_preds.shape[1:],dtype=torch.int)
    cat_num = [len(cat) for cat in label_names]
    if rule is not None: rule_dict = parse_combine_rule(rule)
    for mic_idx in range(mic_preds.shape[1]):
        for cat_idx in range(mic_preds.shape[2]):
            cat_preds = mic_preds[:,mic_idx,cat_idx]
            cat_confs = mic_pred_confidence[:,mic_idx,sum(cat_num[:cat_idx]):sum(cat_num[:cat_idx+1])]
            if (cat_preds == cat_preds[0]).all(): # check all elements are the same
                combined_preds[0,mic_idx,cat_idx] = cat_preds[0] # set to the consensus element if all are same
                combined_preds[1,mic_idx,cat_idx] = find_max_catidx(cat_preds[0],cat_preds,cat_confs)
            else:
                if rule is None: # equivalent to rule 0,111,010,222:2,111,010 or 0,111,010,222:1,000:2,111,010:3,000
                    if cat_idx == 0: # film
                        if 1 in cat_preds: # contains minor_film
                            combined_preds[0,mic_idx,cat_idx] = 1 # set to minor_film
                            combined_preds[1,mic_idx,cat_idx] = find_max_catidx(1,cat_preds,cat_confs)
                        elif 0 in cat_preds: # contains no_film
                            combined_preds[0,mic_idx,cat_idx] = 1 # set to minor_film
                            combined_preds[1,mic_idx,cat_idx] = find_max_catidx(0,cat_preds,cat_confs) # use confidence of no_film
                        else: # major_film and film
                            combined_preds[0,mic_idx,cat_idx] = 2 # set to major_film
                            combined_preds[1,mic_idx,cat_idx] = find_max_catidx(2,cat_preds,cat_confs)
                    elif cat_idx == 2: # crystalline
                        if 1 in cat_preds: # contains minor_crystalline
                            combined_preds[0,mic_idx,cat_idx] = 1  # set to minor_crystalline
                            combined_preds[1,mic_idx,cat_idx] = find_max_catidx(1,cat_preds,cat_confs)
                        else: # not_crystalline and major_crystalline
                            combined_preds[0,mic_idx,cat_idx] = 1 # set to minor_crystalline
                            combined_preds[1,mic_idx,cat_idx] = find_max_catidx(0,cat_preds,cat_confs) # use confidence of not_crystalline
                    else: # drift and contamination, binary catagorization
                        combined_preds[0,mic_idx,cat_idx] = 0 # set to minor
                        combined_preds[1,mic_idx,cat_idx] = find_max_catidx(0,cat_preds,cat_confs) # use confidence of minor
                else: # combine using rule provided
                    if cat_idx in rule_dict:
                        for rule_label in rule_dict[cat_idx]:
                            if rule_label[0] in cat_preds:
                                combined_preds[0,mic_idx,cat_idx] = rule_label[1]
                                combined_preds[1,mic_idx,cat_idx] = find_max_catidx(rule_label[2],cat_preds,cat_confs)
                                break
                    else: # binary catagorization with rule 000
                        combined_preds[0,mic_idx,cat_idx] = 0 # minor
                        combined_preds[1,mic_idx,cat_idx] = find_max_catidx(0,cat_preds,cat_confs) # use confidence of minor
    return combined_preds