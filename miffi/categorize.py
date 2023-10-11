"""
Categorize micrographs based on inference result with default labels
"""

import logging
import argparse
from pathlib import Path
import datetime
import starfile
import numpy as np
import yaml
import torch
from .utils import load_pkl
from .parameters import DEFAULT_LABEL_NAMES, CATEGORY_DEFAULT, CATEGORY_ALL, CATEGORY_GOOD_PREDICTIONS, CONF_SPLIT_NAMES

logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument(
        '-t',
        '--out-type',
        default='list',
        help="Type of the output. Available options: list (default), star, cs"
    )
    parser.add_argument(
        '-i',
        '--inference-result',
        type=Path,
        required=True,
        help="Path to the inference result file"
    )
    parser.add_argument(
        '--star',
        type=Path,
        default=None,
        help="Path to the original star file, only applicable when outputting star files"
    )
    parser.add_argument(
        '--cs',
        type=Path,
        default=None,
        help="Path to the original cs file, only applicable when outputting cs files"
    )
    parser.add_argument(
        '--ptcs',
        type=Path,
        default=None,
        help="Path to the original passthrough cs file, only applicable when outputting cs files"
    )
    parser.add_argument(
        '--csg',
        type=Path,
        default=None,
        help="Path to the original csg file, only applicable when outputting cs files"
    )
    parser.add_argument(
        '-o',
        '--outdir',
        type=Path,
        default=Path.cwd(),
        help="Path to the directory for outputting files",
    )
    parser.add_argument(
        '--outname',
        default=None,
        help="Prefixes to add to the names of the output files. If not provided, outname from the inference step will be used.",
    )
    parser.add_argument(
        '--sb',
        action='store_true',
        help="Output all singly bad predictions into individual categories",
    )
    parser.add_argument(
        '--sc',
        action='store_true',
        help="Split outputs of good and singly bad predictions based on confidence of the prediction",
    )
    parser.add_argument(
        '--gc',
        type=float,
        default=0.7,
        help="Confience cutoff for good predictions. Good prediction with a confidence lower than cutoff will be categoraized as low confidence good prediction",
    )
    parser.add_argument(
        '--bc',
        type=float,
        default=0.7,
        help="Confience cutoff for bad predictions. Bad prediction with a confidence lower than cutoff will be categoraized as low confidence bad prediction",
    )
    parser.add_argument(
        '--bci',
        type=str,
        default=None,
        help="Confience cutoff for bad predictions of each individual categories. Default is using the same cutoff for all categories with the value from --bc argument. Input string should be four float numbers seperated by comma (e.g. 0.8,0.9,0.8,0.9). Order of categories is film, drift, crystalline, contamination.",
    )

def load_yaml(file):
    with open(file,'r') as f:
        return yaml.safe_load(f)

def write_yaml(obj,file):
    with open(file,'w+') as f:
        yaml.dump(obj, f, default_flow_style=False)

def write_cs(cs,file):
    with open(file,'wb') as f:
        np.save(f,cs)

def get_pred_conf(inference_result, mic_idx, cat_idx, pred_idx, cat_idx_to_idx):
    high_conf_idx = inference_result['combined_preds'][1][mic_idx][cat_idx].item()
    return inference_result['mic_pred_confidence'][high_conf_idx][mic_idx][cat_idx_to_idx[cat_idx][pred_idx]]

def list_file_writer(path,cat_mic_list):
    with open(path,'w+') as f:
        for item in cat_mic_list:
            f.write(f'{item}\n')

def star_file_writer(path,cat_mic_list,star):
    star_list = {}
    for key in star:
        if key != 'micrographs':
            star_list[key] = star[key]
        else:
            star_list[key] = star[key][star[key]['rlnMicrographName'].isin(cat_mic_list)]
    assert len(star_list['micrographs']) > 0, f"Micrograph paths are not matching to those in the star file, make sure the correct star file is used"
    starfile.write(star_list, path.with_suffix('.star'))

def cs_file_writer(path,cat_mic_list,cs,ptcs,csg):
    num_mic = len(cat_mic_list)
    
    if 'micrograph_blob/path' in cs.dtype.names:
        cs_to_match = cs
    elif 'micrograph_blob/path' in ptcs.dtype.names:
        cs_to_match = ptcs
    else:
        raise Exception("'micrograph_blob/path' entry not found in either cs file or passthrough cs file")
        
    matching_idx = [idx for idx, pathi in enumerate(cs_to_match['micrograph_blob/path']) if pathi.decode('utf-8') in cat_mic_list]
    assert len(matching_idx) > 0, f"Micrograph paths are not matching to those in the cs file, make sure that the correct cs files are used"
    
    matching_cs = cs[matching_idx]
    matching_ptcs = ptcs[matching_idx]
    
    cs_outname = path.with_suffix('.cs')
    ptcs_outname = path.with_name(f'{path.stem}_passthrough.cs')
    csg_outname = path.with_suffix('.csg')

    cs_fields = list(set([entry.split('/')[0] for entry in cs.dtype.names]))
    ptcs_fields = list(set([entry.split('/')[0] for entry in ptcs.dtype.names]))
    field_file_dict = {}
    for field in ptcs_fields:
        field_file_dict[field] = ptcs_outname.name
    for field in cs_fields:
        field_file_dict[field] = cs_outname.name
    
    csg_new = {}
    csg_new['create'] = datetime.datetime.today()
    csg_new['group'] = {'description': '',
                        'name': path.stem,
                        'title': path.stem.replace('_',' ').capitalize(),
                        'type': 'exposure'}
    csg_new['results'] = {}
    for key in csg['results']:
        if key in field_file_dict.keys():
            csg_new['results'][key] = {}
            csg_new['results'][key]['metafile'] = f'>{field_file_dict[key]}'
            csg_new['results'][key]['num_items'] = num_mic
            csg_new['results'][key]['type'] = csg['results'][key]['type']
        else:
            logger.warning(f"key {key} not found in either cs or ptcs files, skip writing")
    csg_new['version'] = csg['version']

    write_cs(matching_cs,cs_outname)
    write_cs(matching_ptcs,ptcs_outname)
    write_yaml(csg_new,csg_outname)

def file_writer(type, path, cat_mic_list, star=None, cs=None, ptcs=None, csg=None):
    if type == 'list':
        list_file_writer(path,cat_mic_list)
    elif type == 'star':
        star_file_writer(path,cat_mic_list,star)
    elif type == 'cs':
        cs_file_writer(path,cat_mic_list,cs,ptcs,csg)
    else:
        raise NotImplementedError(f"file writer with type {type} is not implemented!")

def main(args):
    inference_result = load_pkl(args.inference_result)
    if args.outname is None:
        outname = inference_result['args'].outname
    else:
        outname = args.outname
    
    file_handler = logging.FileHandler(filename=args.outdir/f'{outname}.categorize.log'.strip('.'))
    file_handler.setFormatter(logging.root.handlers[0].formatter)
    logger.addHandler(file_handler)
    
    assert args.out_type.lower() in ['list','star','cs'], f"Unsupported output type {args.out_type}"
    logger.info(f"Writing result as {args.out_type.lower()} files")

    ori_star = starfile.read(args.star) if args.star is not None else None
    if args.out_type.lower() == 'star':
        assert ori_star is not None, f"Outputting star file requires providing the original .star file"

    ori_cs = np.load(args.cs) if args.cs is not None else None
    ori_ptcs = np.load(args.ptcs) if args.ptcs is not None else None
    ori_csg = load_yaml(args.csg) if args.csg is not None else None
    if args.out_type.lower() == 'cs':
        assert all(file is not None for file in [ori_cs, ori_ptcs, ori_csg]), f"Outputting cs files requires providing the original .cs, passthrough .cs, and .csg files"
        assert len(ori_cs) == len(ori_ptcs), f"Provided .cs file contains different number of micrographs as comapred to the passthrough .cs file, make sure that the correct files are used"

    if args.bci is not None:
        bc_indi = list(map(float,args.bci.split(',')))
        assert len(bc_indi) == 4, "The number of individual bad confidence cutoff input is not 4! Check your --bci input."
    else:
        bc_indi = list([args.bc])*4
    
    total_cat = len(DEFAULT_LABEL_NAMES)
    cat_num = [len(cat) for cat in DEFAULT_LABEL_NAMES]
    cat_idx_to_idx = [list(range(sum(cat_num[:cat_idx]),sum(cat_num[:cat_idx+1]))) for cat_idx in range(len(cat_num))]
    cat_good_pred = CATEGORY_GOOD_PREDICTIONS

    mic_list = [str(mic_path) for mic_path in inference_result['mic_list']]
    mic_category = {cate:[[],[]] for cate in CATEGORY_ALL}
    for mic_idx in range(len(mic_list)):
        cat_pred_is_good = [inference_result['combined_preds'][0][mic_idx][cat_idx].item() in cat_good_pred[cat_idx] for cat_idx in range(total_cat)]
        if all(cat_pred_is_good):
            pred_conf = [sum(get_pred_conf(inference_result, mic_idx, cat_idx, pred_idx, cat_idx_to_idx) for pred_idx in cat_good_pred[cat_idx]) 
                         for cat_idx in range(total_cat)]
            mic_category['good'][int(any(conf<args.gc for conf in pred_conf))].append(mic_list[mic_idx])
        elif cat_pred_is_good.count(False)==1:
            bad_cat_idx = cat_pred_is_good.index(False)
            pred_conf = sum(get_pred_conf(inference_result, mic_idx, bad_cat_idx, pred_idx, cat_idx_to_idx) 
                            for pred_idx in range(cat_num[bad_cat_idx]) if pred_idx not in cat_good_pred[bad_cat_idx])
            mic_category['bad_single'][int(pred_conf<args.bc)].append(mic_list[mic_idx])
            if bad_cat_idx == 0:
                mic_category['bad_film'][int(pred_conf<bc_indi[bad_cat_idx])].append(mic_list[mic_idx])
            elif bad_cat_idx == 1:
                mic_category['bad_drift'][int(pred_conf<bc_indi[bad_cat_idx])].append(mic_list[mic_idx])
            elif bad_cat_idx == 2:
                if inference_result['combined_preds'][0][mic_idx][bad_cat_idx].item() == 1:
                    mic_category['bad_minor_crystalline'][int(pred_conf<bc_indi[bad_cat_idx])].append(mic_list[mic_idx])
                else:
                    pred_conf = get_pred_conf(inference_result, mic_idx, bad_cat_idx, 2, cat_idx_to_idx)
                    mic_category['bad_major_crystalline'][int(pred_conf<bc_indi[bad_cat_idx])].append(mic_list[mic_idx])
            elif bad_cat_idx == 3:
                mic_category['bad_contamination'][int(pred_conf<bc_indi[bad_cat_idx])].append(mic_list[mic_idx])
        else:
            mic_category['bad_multiple'][0].append(mic_list[mic_idx])

    if args.sb:
        category_to_write = CATEGORY_ALL
        logger.info("Writing all four singly bad prediciton categories")
    else:
        category_to_write = CATEGORY_DEFAULT
    
    if args.sc:
        logger.info("Splitting categories based on confidence scores")
        logger.info(f"Good prediction high confidence cutoff is {args.gc}")
        logger.info(f"Bad prediction high confidence cutoff is {args.bc}")
        if args.bci is not None:
            logger.info(f"Bad prediction high confidence cutoff for each individual category is {' '.join([str(conf) for conf in bc_indi])}")
    
    conf_split_names = CONF_SPLIT_NAMES
    
    logger.info("Number of micrographs in categories:")
    line_size = 40
    for category in category_to_write:
        if category == 'bad_multiple' or not args.sc:
            all_mic_this_category = sum(mic_category[category],[])
            category_len = len(all_mic_this_category)
            logger.info(f"{category}{'.'*(max(0,line_size-len(category)))}{category_len}")
            if category_len > 0:
                file_writer(args.out_type.lower(),args.outdir/f'{outname}_{category}'.strip('_'),all_mic_this_category,
                            ori_star,ori_cs,ori_ptcs,ori_csg)
        else:
            for split_idx in range(len(mic_category[category])):
                category_split_len = len(mic_category[category][split_idx])
                category_split_name = f'{category}_{conf_split_names[split_idx]}'
                logger.info(f"{category_split_name}{'.'*(max(0,line_size-len(category_split_name)))}{category_split_len}")
                if category_split_len > 0:
                    file_writer(args.out_type.lower(),args.outdir/f'{outname}_{category_split_name}'.strip('_'),mic_category[category][split_idx],
                                ori_star,ori_cs,ori_ptcs,ori_csg)

    logger.info('All done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())