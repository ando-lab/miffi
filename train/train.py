"""
Main script for training
"""

import logging
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from datasets import MicDataset, mic_transforms
from convnext_utils import LayerDecayValueAssigner, create_optimizer, cosine_scheduler
import time
from time import localtime, strftime
import pickle

logging.basicConfig(format='(%(levelname)s|%(filename)s|%(asctime)s) %(message)s', level=logging.INFO, 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument(
        '-t',
        '--label-dict-train',
        type=Path,
        help="Path to the label dictionary for training data"
    )
    parser.add_argument(
        '-v',
        '--label-dict-val',
        type=Path,
        help="Path to the label dictionary for validation data"
    )
    parser.add_argument(
        '-d',
        '--datadir',
        type=Path,
        default=Path.cwd(),
        help="Path to directory containing data (will be prepended to path obtained from label dict)"
    )
    parser.add_argument(
        '-o',
        '--outdir',
        type=Path,
        default=Path.cwd(),
        help="Path to directory for outputting training results",
    )
    parser.add_argument(
        '--outname',
        default='',
        help="Prefixes to add to the name of the outputs",
    )
    parser.add_argument(
        '--no-pretrain',
        action='store_true',
        help="Don not initialize with pretrained model",
    )
    parser.add_argument(
        '--no-ps',
        action='store_true',
        help="Don not use power spectrum in training",
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help="Number of workers to use for dataloader",
    )
    parser.add_argument(
        '--model-to-train',
        default='convnext_small.in12k',
        help="CNN model to obtain from timm for training",
    )
    parser.add_argument(
        '--opt',
        default='adamw',
        help="Optimizer to use",
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        help="Base learning rate to use",
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-8,
        help="Weight decay to use",
    )
    parser.add_argument(
        '--layer-decay',
        type=float,
        default=0.9,
        help="Learning rate layer decay ratio to use. Only works with ConvNeXt models.",
    )
    parser.add_argument(
        '--min-lr',
        type=float,
        default=1e-6,
        help="Minimum learning rate to use",
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=0,
        help="Number of warm up epochs",
    )
    parser.add_argument(
        '-n',
        '--num-epochs',
        type=int,
        default=10,
        help="Number of epochs to train",
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=32,
        help="Batch size to use for dataloader",
    )
    parser.add_argument(
        '--label-names',
        type=Path,
        default=None,
        help="Path to a pkl file that contains the list of label names",
    )
    parser.add_argument(
        '--sd',
        type=Path,
        default=None,
        help="Path to a state dict file for initializing the model",
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        default=None,
        type=int,
        help="Frequency to save intermediate state dict (save every N epochs). Default is only saving the final state dict.",
    )
    parser.add_argument(
        '--use-cpu',
        action='store_true',
        help="Use CPU for training",
    )
    parser.add_argument(
        '-g',
        '--gpu-id',
        type=int,
        default=0,
        help="GPU id for training",
    )

def output_to_pred(outputs, label_names, device):
    preds = torch.zeros(outputs.shape[0],len(label_names),dtype=torch.int, device=device)
    cat_num = [len(cat) for cat in label_names]
    for idx in range(len(label_names)):
        _, max_loc = torch.max(outputs[:,sum(cat_num[:idx]):sum(cat_num[:idx+1])],1)
        preds[:,idx] = max_loc
    return preds

def combined_accu(preds, labels, label_names, device):
    indi_accu = torch.zeros(len(label_names), dtype=torch.float32, device=device)
    for idx in range(4):
        indi_accu[idx] = torch.sum(preds[:,idx] == labels[:,idx])
    accu = torch.sum(indi_accu)/len(label_names)
    return accu, indi_accu

def combined_loss(outputs, labels, label_names, device):
    loss = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
    indi_loss = torch.zeros(len(label_names), dtype=torch.float32, device=device)
    cat_num = [len(cat) for cat in label_names]
    for idx in range(len(label_names)):
        indi_loss_tmp = F.cross_entropy(outputs[:,sum(cat_num[:idx]):sum(cat_num[:idx+1])],labels[:,idx].type(torch.long))
        loss = loss + indi_loss_tmp
        indi_loss[idx] = indi_loss_tmp
    return loss, indi_loss

def train_model(model, dataloaders, dataset_sizes, criterion, accuracy, optimizer, lr_schedule, num_epochs, label_names, device, checkpoint, outdir, outname, time_stamp):
    since = time.time()
    loss_by_epoch = {'train':[], 'val':[]}
    acc_by_epoch = {'train':[], 'val':[]}

    for epoch in range(num_epochs):
        logger.info("-" * 10)
        logger.info(f"Epoch {epoch}/{num_epochs - 1}")
        logger.info("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_loss_indi = torch.zeros(len(label_names), dtype=torch.float32)
            running_accu = 0.0
            running_accu_indi = torch.zeros(len(label_names), dtype=torch.float32)

            for num_batch, batch in enumerate(dataloaders[phase]):
                if num_batch%100 == 0: logger.info(f"batch {num_batch} / {len(dataloaders[phase]) - 1}")
                inputs = batch['data'].to(device)
                labels = batch['label'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # set learning rates
                if phase == 'train':
                    it = epoch * len(dataloaders['train']) + num_batch
                    for i, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = lr_schedule[it] * param_group['lr_scale']
                    if num_batch%10 == 0: logger.debug(f"current learning rate: {lr_schedule[it]}")
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = output_to_pred(outputs, label_names, device)
                    accu, indi_accu = accuracy(preds, labels, label_names, device)
                    loss, indi_loss = criterion(outputs, labels, label_names, device)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss_indi += indi_loss.detach().cpu()
                running_accu += accu.item()
                running_accu_indi += indi_accu.detach().cpu()
            

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_indi = running_loss_indi / dataset_sizes[phase]
            loss_by_epoch[phase].append([epoch_loss,epoch_loss_indi])
            epoch_acc = running_accu / dataset_sizes[phase]
            epoch_accu_indi = running_accu_indi / dataset_sizes[phase]
            acc_by_epoch[phase].append([epoch_acc,epoch_accu_indi])

            logger.info(f"{phase} Loss: {epoch_loss:.4f} {epoch_loss_indi} Acc: {epoch_acc:.4f} {epoch_accu_indi}")
            
            if (checkpoint is not None and epoch%checkpoint == 0) or (epoch == num_epochs - 1):
                torch.save(model.state_dict(), outdir/f'{outname}_state_dict_{time_stamp}.{epoch}.pt'.strip('_'))
    
    training_stats = {'loss':loss_by_epoch, 'accu':acc_by_epoch}
    
    time_elapsed = time.time() - since
    logger.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logger.info(f"Final val Acc: {epoch_acc:4f}")
    
    return model, training_stats

def main(args):
    time_stamp = strftime("%m_%d_%H_%M", localtime())
    file_handler = logging.FileHandler(filename=args.outdir/f'{args.outname}_train_{time_stamp}.log'.strip('_'))
    file_handler.setFormatter(logging.root.handlers[0].formatter)
    logger.addHandler(file_handler)
    
    with open(args.label_dict_train,'rb') as f:
        label_dict_train = pickle.load(f)
    with open(args.label_dict_val,'rb') as f:
        label_dict_val = pickle.load(f)
    if args.label_names is not None:
        with open(args.label_names,'rb') as f:
            label_names = pickle.load(f)
    else:
        label_names = [['no_film','minor_film','major_film','film'],
                       ['no_crack_drift','crack_drift_empty'],
                       ['not_crystalline','minor_crystalline','major_crystalline'],
                       ['minor_contamination','major_contamination']]

    dataset_train = MicDataset(label_dict_train, args.datadir, label_names, mic_transforms(args.no_ps)['train'])
    dataset_val = MicDataset(label_dict_val, args.datadir, label_names, mic_transforms(args.no_ps)['val'])
    datasets = {'train':dataset_train, 'val':dataset_val}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) 
                   for x in ['train', 'val']}
    
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu_id}')
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available for torch! Using CPU for training.")

    if args.no_ps:
        num_chans = 1
    else:
        num_chans = 2
    model_ini = timm.create_model(args.model_to_train, pretrained=not args.no_pretrain, 
                                  in_chans=num_chans, num_classes=len(sum(label_names,[]))).to(device)
    if args.sd is not None:
        logger.info("Load provided state dict into model")
        model_ini.load_state_dict(torch.load(args.sd, map_location=device))

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None
    
    optimizer = create_optimizer(args, model_ini, skip_list=None, 
                                 get_num_layer=assigner.get_layer_id if assigner is not None else None, 
                                 get_layer_scale=assigner.get_scale if assigner is not None else None)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.num_epochs, len(dataloaders['train']), args.warmup_epochs)

    model_final, training_stats = train_model(model_ini, dataloaders, dataset_sizes, combined_loss, combined_accu, optimizer, 
                                              lr_schedule, args.num_epochs, label_names, device, args.checkpoint, args.outdir, args.outname, time_stamp)

    training_stats['args'] = args
    with open(args.outdir/f'{args.outname}_training_stats_{time_stamp}.pkl'.strip('_'),'wb') as f:
        pickle.dump(training_stats,f)
    
    logger.info("All done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())