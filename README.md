# `Miffi: Cryo-EM micrograph filtering utilizing Fourier space information`

BioRxiv: https://doi.org/10.1101/2023.12.08.570849

## Repository Contents

- `miffi/` - Main Miffi package
- `train/` - Model training scripts (independent from Miffi package)
- `empiar_test_set_labels/` - Labels for EMPIAR test sets

## Installation

First create a conda virtual enviroment for Miffi:

```
conda create -n miffi python=3.9
conda activate miffi
```

### Install from local copy

Obtain source code for Miffi using `git clone` (downloading zip from GitHub will lead to issue due to missing .git metadata):

```
git clone https://github.com/ando-lab/miffi.git
```

Then change directory into the source code folder and install using pip:

```
cd miffi
pip install .
```

If you need to install Pytorch with a lower CUDA version due to compatibility issue with your GPU driver, follow the instruction on Pytorch website (https://pytorch.org/get-started/locally/) and install appropriate versions of Pytorch and CUDA using conda or pip before doing `pip install .` in the step above. Note that full Miffi models were generated with TorchScript in Pytorch version 1.13.1, which may not work in older Pytorch versions. However, state dicts may be used instead, or could be converted to full models using older Pytorch versions.

## Usage

The following sections outline basic usage of the Miffi package. To see all available arguments for different commands, refer to `--help`.

### Download Miffi models

Full Miffi models and state dicts are hosted on Zenodo (https://doi.org/10.5281/zenodo.8342009). Programatic download for full models is available as part of the Miffi package. To download the current default model:

```
miffi download -d path/to/download/dir
```

To download all available full models, use `--all` argument. A specific model can be downloaded by using `--model-name model_to_download`. Currently available models:

- `miffi_v1` - Model trained with finetuning and power spectrum input. (Default model)
- `miffi_no_ps_v1` - Model trained with finetuning but without power spectrum input. May be useful for filtering micrographs with pixel sizes larger than 1.5 Å.
- `miffi_no_pretrain_no_ps_v1` - Model trained without finetuning or power spectrum input. (not recommended for anything, for validation purpose only)

### Perform inference on micrographs

Micrograph input can be provided as folder and wildcard to match (`--micdir micrograph_dir -w wildcard`), or can be obtained from a file (a star file, a cryosparc csg file, a plain text list, or a pkl file containing a list). Example for performing inference on micrographs from a csg file:

```
miffi inference -f path/to/csgfile -d path/to/data/dir --outname output_name \
                -g gpu_id_to_use -m path/to/model/dir
```

Details on arguments used above:

- `-f` Path to the file specifying input micrograph list (csg file in this case).
- `-d` Path to the data directory, which will be prepended to the micrograph path found in the input file. This is needed when micrograph path in the file is not absolute path. For star and csg files, this is usually the path to the root directory of the processing project.
- `--outname` Name to be prepended to the output result file.
- `-g` GPU id to use for inference.
- `-m` Path to the directory containing full model files.

Output containing prediction confidence scores will be saved as a dictionary in a pkl file.

### Categorize micrographs

After inference, micrographs can be split into multiple categories based on inference results. Output format can be one of the following: plain text list files, star files, or cs files. To output star and cs files, the original star file or the original csg file needs to be provided. Example for categorizing and outputing cs files:

```
miffi categorize -t cs -i path/to/inference/result --csg path/to/original/csgfile \
                 --sb --sc
```

Details on arguments used above:

- `-t` Type of the output, can be one of the following: `list`, `star`, `cs`.
- `-i` Path to the inference result file.
- `--csg` Path to the original csg file, which is necessary for outputting cs files. To output star files, path to the original star file needs to be provided through `--star`.
- `--sb` Split all singly bad micrographs into individual categories. If this argument is not used, only film and minor crystalline categories will be written.
- `--sc` Split all categories into high and low confidence based on a cutoff value. Default confidence cutoff is 0.7, but can be configured through `--gc` for good predictions and `--bc` for bad predictions. Argument `--bci` can be used to set confidence cutoff for bad predictions of each category individually.

To load output cs files into cryosparc, use "Import Result Group" job and input the output csg file of the corresponding category.

### Write png files for micrographs

Miffi package also includes a tool for writing micrographs with their power spectrum as png files for easy visual inspection. The file input options are idential to inference. Example for writing png files from a star file:

```
miffi write_png -f path/to/starfile -d path/to/data/dir -o path/to/output/dir \
                -q -n number_of_threads
```

Details on arguments used above:

- `-f` Path to the file containing input micrograph list.
- `-d` Path to the data directory.
- `-o` Path to the output directory.
- `-q` Do not preprocess (split into square micrographs and downasmple) before writing png files.
- `-n` Number of threads to use for writing png files.

## Train new models

See `README.md` file in the `train` folder.
