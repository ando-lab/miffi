# Training scripts for miffi models

This folder contains scripts used to train the miffi models. These scripts are independent from the miffi package. Refer to `--help` for all available arguments for each script.

## Preprocess

Micrographs used in training need to be first preprocessed using `preprocess.py`. This process include split or crop into square micrographs and then Fourier crop to a set pixel size (default: 1.5 Å). Output include preprocessed mrc files along with png files for labeling. Example usage:

```
python preprocess.py path/to/micrograph/dir -w wildcard -o path/to/output/dir \
                     -n number_of_threads
```

## Labeling

Jupyter notebook `interactive_labeling.ipynb` provides GUI-based labeling of preprocessed micrographs using ipywidgets. Input is the preprocessed png files, and output is the label dictionary saved in a pkl file.

## Training

Model training is done using `train.py`. Labels are expected as dictionaries of which the keys are path to the micrograph mrc files while the values are the corresponding labels. Example usage:

```
python train.py -t path/to/training/labels -v path/to/validation/labels \
                --outname output_name --lr learning_rate --layer-decay layer_decay_ratio \
                --weight-decay weight_decay --batch-size batch_size
                --num-workers number_of_workers_for_dataloader \
                --num-epochs number_of_epochs_to_train -g gpu_id_to_use
```

Outputs include trained state dict, training stats, and the training log file.
