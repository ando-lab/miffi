"""
Inference micrographs
"""

import copy
import logging
import warnings
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .datasets import MicDataset, mic_transforms
from .utils import load_pkl, save_pkl, get_miclist, get_preprocess_param, combine_preds
import time
from time import localtime, strftime
from .parameters import DEFAULT_LABEL_NAMES, AVAILABLE_MODELS
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
        help="Path to directory for outputting the inference result file",
    )
    parser.add_argument(
        '--outname',
        default='',
        help="Prefixes to add to the name of the inference result file",
    )
    parser.add_argument(
        '-p',
        '--pixel-size',
        type=float,
        default=None,
        help="Pixel size of input micrographs in the unit of angstrom (overwrites pixel size obtained from micrograph mrc files)",
    )
    parser.add_argument(
        '--downsample-pixel-size',
        type=float,
        default=1.5,
        help="Pixel size to downsample the micrograph to (in the unit of angstrom)",
    )
    parser.add_argument(
        '-s',
        '--max-split',
        type=int,
        default=None,
        help="Maximum number of micrographs to split non-square micrographs into (starting from top/left)",
    )
    parser.add_argument(
        '--use-cpu',
        action='store_true',
        help="Use CPU for inference",
    )
    parser.add_argument(
        '-g',
        '--gpu-id',
        type=int,
        default=0,
        help="GPU id for inference",
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=8,
        help="Batch size to use for dataloader",
    )
    parser.add_argument(
        '-n',
        '--num-workers',
        type=int,
        default=8,
        help="Number of workers to use for dataloader",
    )
    parser.add_argument(
        '-m',
        '--model-folder',
        type=Path,
        help="Path to the folder containing miffi models. This input is ignored if the path to a model file is provided.",
    )
    parser.add_argument(
        '--model-name',
        default='miffi_v1',
        help=f"Miffi model to use for inference. This input is ignored if the path to a model file is provided. Available options: {', '.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument(
        '--fm',
        type=Path,
        default=None,
        help="Path to a full model in TorchScript format to be used in inference. Overwrites miffi model inputs.",
    )
    parser.add_argument(
        '--sd',
        type=Path,
        default=None,
        help="Path to a state dict to be loaded into the model. Overwrites miffi model inputs and full model inputs.",
    )
    parser.add_argument(
        '--model-to-load',
        default='convnext_small',
        help="Model to obtain from timm to load the state dict into",
    )
    parser.add_argument(
        '--no-ps',
        action='store_true',
        help="Do not include power spectrum during preprocessing. Only use with models trained without power spectrum.",
    )
    parser.add_argument(
        '--label-names',
        type=Path,
        default=None,
        help="Path to a pkl file that contains the list of label names, which needs to be consistent with the model used",
    )
    parser.add_argument(
        '--combine-rule',
        type=str,
        default=None,
        help="A string specifying the rule for combining predictions of splits from a mirograph",
    )

def get_label_names(arg_label_names):
    if arg_label_names is not None:
        label_names = load_pkl(arg_label_names)
    else:
        label_names = DEFAULT_LABEL_NAMES
    return label_names

def get_device(use_cpu, gpu_id):
    if use_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available for torch! Using CPU for inference")
    return device

def output_to_pred(outputs, label_names, device):
    preds = torch.zeros(outputs.shape[0],len(label_names),dtype=torch.int, device=device)
    cat_num = [len(cat) for cat in label_names]
    for idx in range(len(label_names)):
        _, max_loc = torch.max(outputs[:,sum(cat_num[:idx]):sum(cat_num[:idx+1])],1)
        preds[:,idx] = max_loc
    return preds

def outputs_to_confidence(mic_outputs, label_names):
    confs = torch.zeros(*mic_outputs.shape,dtype=torch.float32)
    cat_num = [len(cat) for cat in label_names]
    for split_idx in range(mic_outputs.shape[0]):
        for mic_idx in range(mic_outputs.shape[1]):
            for cat_idx in range(len(cat_num)):
                confs[split_idx,mic_idx,sum(cat_num[:cat_idx]):sum(cat_num[:cat_idx+1])] = F.softmax(mic_outputs[split_idx,mic_idx,sum(cat_num[:cat_idx]):sum(cat_num[:cat_idx+1])], dim=0)
    return confs

def load_model(model_folder, model_name, fm, sd, model_to_load, no_ps, label_names, device):
    if sd is not None:
        logger.info(f"Use provided state dict and load into {model_to_load} model")
        import timm
        if no_ps:
            num_chans = 1
        else:
            num_chans = 2
        model = timm.create_model(model_to_load, pretrained=False, in_chans=num_chans, num_classes=len(sum(label_names,[]))).to(device)
        model.load_state_dict(torch.load(sd, map_location=device))
    elif fm is not None:
        logger.info("Use provided full model")
        model = torch.jit.load(fm, map_location=device)
    else:
        assert model_name in AVAILABLE_MODELS, f"Specified model name {model_name} is not supported!"
        logger.info(f"Use miffi model {model_name}")
        model_path = model_folder / f'{model_name}.pt'
        assert model_path.exists(), f"Model file {model_name}.pt cannot be found in {model_folder}"
        model = torch.jit.load(model_path, map_location=device)
    return model

def inference_mic(model, dataloader, label_names, device, num_mic, batch_size):
    since = time.time()
    mic_outputs = []
    mic_preds = []

    with tqdm(total=num_mic, unit='mic', smoothing=0) as pbar:
        for num_batch, batch in enumerate(dataloader):
            inputs = [batch_i.to(device) for batch_i in batch]
            
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    outputs = [model(input) for input in inputs]
                preds = [output_to_pred(output, label_names, device) for output in outputs]
            
                mic_outputs.append(torch.stack(outputs).detach().cpu())
                mic_preds.append(torch.stack(preds).detach().cpu())
            
            pbar.update(min(batch_size,num_mic-num_batch*batch_size))
            
    mic_outputs = torch.cat(mic_outputs,1)
    mic_preds = torch.cat(mic_preds,1)
    time_elapsed = time.time() - since
    logger.info(f"Inference completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
    return mic_outputs, mic_preds

def main(args):
    mic_list = get_miclist(args.miclist_file, args.micdir, args.wildcard)

    label_names = get_label_names(args.label_names)

    no_ps = args.no_ps if (args.fm is not None or args.sd is not None) else 'no_ps' in args.model_name
    
    transpose_mic, resized_small_dim, num_split = get_preprocess_param(args.datapath / mic_list[0], args.pixel_size, 
                                                                       args.downsample_pixel_size, args.max_split, no_ps)

    dataset = MicDataset(mic_list, args.datapath, label_names, 
                         transform=mic_transforms(resized_small_dim, num_split, transpose_mic, no_ps))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = get_device(args.use_cpu, args.gpu_id)

    model = load_model(args.model_folder, args.model_name.lower(), args.fm, args.sd, args.model_to_load, no_ps, label_names, device)
    model.eval()

    logger.info(f"Start inference on {len(mic_list)} micrographs as {len(dataloader)} batches")
    mic_outputs, mic_preds = inference_mic(model, dataloader, label_names, device, len(mic_list), args.batch_size)

    mic_pred_confidence = outputs_to_confidence(mic_outputs, label_names)
    if args.combine_rule is not None: logger.info(f"Combine predictions using provided rule: {args.combine_rule}")
    combined_preds = combine_preds(mic_preds, mic_pred_confidence, label_names, args.combine_rule)

    args_to_save = copy.deepcopy(args)
    args_to_save.func = None
    save_pkl({'combined_preds': combined_preds, 'mic_pred_confidence':mic_pred_confidence,
              'outputs': mic_outputs, 'preds': mic_preds, 'mic_list': mic_list, 'args': args_to_save}, 
             args.outdir/f'{args.outname}_inference_result_{strftime("%m_%d_%H_%M", localtime())}.pkl'.strip('_'))

    logger.info("All done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())