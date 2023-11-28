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
from .utils import load_pkl, save_pkl, load_yaml
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

def write_yaml(obj,file):
    with open(file,'w+') as f:
        yaml.dump(obj, f, default_flow_style=False)

def write_cs(cs,file):
    with open(file,'wb') as f:
        np.save(f,cs)

def get_pred_conf(inference_result, mic_idx, cat_idx, pred_idx, cat_idx_to_idx):
    high_conf_idx = inference_result['combined_preds'][1][mic_idx][cat_idx].item()
    return inference_result['mic_pred_confidence'][high_conf_idx][mic_idx][cat_idx_to_idx[cat_idx][pred_idx]]

def list_file_writer(items):
    for item in items:
        path = item[0]
        cat_mic_list = item[1]
        
        with open(path,'w+') as f:
            for entry in cat_mic_list:
                f.write(f'{entry}\n')

def star_file_writer(items,star):
    star_ori = starfile.read(star)
    assert 'micrographs' in star_ori.keys(), "micrographs entry not found in the provided star file!"
    
    for item in items:
        path = item[0]
        cat_mic_list = item[1]
        
        star_new = {}
        for key in star_ori:
            if key != 'micrographs':
                star_new[key] = star_ori[key]
            else:
                star_new[key] = star_ori[key][star_ori[key]['rlnMicrographName'].isin(cat_mic_list)]
        assert len(star_new['micrographs']) > 0, f"Micrograph paths are not matching to those in the star file, make sure that the correct star file is used"
        starfile.write(star_new, path.with_suffix('.star'))

def cs_file_writer(items,csg):
    csg_ori = load_yaml(csg)
    assert 'micrograph_blob' in csg_ori['results'].keys(), "'micrograph_blob' entry not found in the provided csg file!"
    
    cs_file_name_dict = {key:csg_ori['results'][key]['metafile'].replace('>','') for key in csg_ori['results'].keys()}
    cs_file_name_list = list(set(list(cs_file_name_dict.values())))
    cs_file_dict_to_list = {key:cs_file_name_list.index(cs_file_name_dict[key]) for key in cs_file_name_dict.keys()}
    
    cs_file_path_list = [csg.parent/entry for entry in cs_file_name_list]
    for cs_file_path in cs_file_path_list:
        assert cs_file_path.exists(), f"cs file {cs_file_path.name} specified in the csg file does not exist in the same directory!"
    
    cs_list = [np.load(cs_file_path) for cs_file_path in cs_file_path_list]
    cs_with_mic_list_idx = cs_file_name_list.index(csg_ori['results']['micrograph_blob']['metafile'].replace('>',''))
    cs_with_mic_list = cs_list[cs_with_mic_list_idx]
    assert 'micrograph_blob/path' in cs_with_mic_list.dtype.names, f"'micrograph_blob/path' entry was not found in the cs file {cs_file_name_list[cs_with_mic_list_idx]} specified in the csg file!"

    for item in items:
        path = item[0]
        cat_mic_list = item[1]
        
        num_mic = len(cat_mic_list)
        
        matching_idx = [idx for idx, pathi in enumerate(cs_with_mic_list['micrograph_blob/path']) if pathi.decode('utf-8') in cat_mic_list]
        assert len(matching_idx) > 0, f"Micrograph paths are not matching to those in the cs file specified in csg file, make sure that the correct csg file is used"
        
        matching_cs_list = [cs[matching_idx] for cs in cs_list]
        
        csg_outname = path.with_suffix('.csg')
        cs_outname_list = [path.with_name(f'{path.stem}_{cs_idx}.cs') for cs_idx, cs_file_name in enumerate(cs_file_name_list)]
        
        csg_new = {}
        csg_new['create'] = datetime.datetime.today()
        title = path.stem.replace('_',' ')
        csg_new['group'] = {'description': csg_ori['group']['description'],
                            'name': path.stem,
                            'title': title[:1].upper()+title[1:],
                            'type': csg_ori['group']['type']}
        csg_new['results'] = {}
        for key in csg_ori['results']:
            csg_new['results'][key] = {}
            csg_new['results'][key]['metafile'] = f'>{cs_outname_list[cs_file_dict_to_list[key]].name}'
            csg_new['results'][key]['num_items'] = num_mic
            csg_new['results'][key]['type'] = csg_ori['results'][key]['type']
        csg_new['version'] = csg_ori['version']
        
        write_yaml(csg_new,csg_outname)
        for cs, cs_file_path in zip(matching_cs_list,cs_outname_list):
            write_cs(cs,cs_file_path)

def file_writer(type, items, star=None, csg=None):
    if type == 'list':
        list_file_writer(items)
    elif type == 'star':
        star_file_writer(items,star)
    elif type == 'cs':
        cs_file_writer(items,csg)
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
    
    if args.out_type.lower() == 'star':
        assert args.star is not None, f"Outputting star file requires providing the original .star file"
    
    if args.out_type.lower() == 'cs':
        assert args.csg is not None, f"Outputting cs files requires providing the original .csg file"
    
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
            mic_category['bad_single'][int(pred_conf<bc_indi[bad_cat_idx])].append(mic_list[mic_idx])
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
        if args.bci is None:
            logger.info(f"Bad prediction high confidence cutoff is {args.bc}")
        else:
            logger.info(f"Bad prediction high confidence cutoff for each individual category is {' '.join([str(conf) for conf in bc_indi])}")
    
    conf_split_names = CONF_SPLIT_NAMES
    
    logger.info("Number of micrographs in categories:")
    items_to_write = []
    category_dict = {}
    line_size = 40
    for category in category_to_write:
        if category == 'bad_multiple' or not args.sc:
            all_mic_this_category = sum(mic_category[category],[])
            category_len = len(all_mic_this_category)
            logger.info(f"{category}{'.'*(max(0,line_size-len(category)))}{category_len}")
            if category_len > 0:
                items_to_write.append([args.outdir/f'{outname}_{category}'.strip('_'),all_mic_this_category])
                category_dict[category] = all_mic_this_category
        else:
            for split_idx in range(len(mic_category[category])):
                category_split_len = len(mic_category[category][split_idx])
                category_split_name = f'{category}_{conf_split_names[split_idx]}'
                logger.info(f"{category_split_name}{'.'*(max(0,line_size-len(category_split_name)))}{category_split_len}")
                if category_split_len > 0:
                    items_to_write.append([args.outdir/f'{outname}_{category_split_name}'.strip('_'),mic_category[category][split_idx]])
                    category_dict[category_split_name] = mic_category[category][split_idx]
    
    file_writer(args.out_type.lower(),items_to_write,args.star,args.csg)
    
    save_pkl(category_dict, args.outdir/f'{outname}_category_dict.pkl')
    
    logger.info('All done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())