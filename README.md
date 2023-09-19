# `miffi: cryo-EM micrograph filtering utilizing Fourier space information`

## Repository Contents

- `miffi/` - Main miffi package
- `train/` - Model training scripts (independent from miffi package)
- `empiar_test_set_labels/` - Labels for EMPIAR test sets

## Installation

First create a conda virtual enviroment for miffi:

```
conda create -n miffi python=3.9
conda activate miffi
```

### Install from local copy

Install pytorch and torchvision compiled with appropriate cudatoolkit version for your GPU. The supported CUDA version for your GPU/driver can usually be found with `nvidia-smi` (11.7 in this case).

```
conda install "pytorch>=1.13.1" torchvision cudatoolkit=11.7 -c pytorch -c nvidia -c conda-forge
```

Note that full miffi models were generated with TorchScript in pytorch version 1.13.1, which may not work in older pytorch versions. However, state dicts may be used instead, or could be converted to full models using older pytorch versions.

Obtain source code for miffi either by downloading from github website, or using git:

```
git clone https://github.com/ando-lab/miffi.git
```

Then change directory into the source code folder and install using pip:

```
cd miffi
pip install .
```

## Usage

The following sections outline basic usage of the miffi package. To see all available arguments for different commands, refer to `--help`.

### Download miffi models

Full miffi models and state dicts are hosted on Zenodo (https://doi.org/10.5281/zenodo.8342009). Programatic download for full models is available as part of the miffi package. To download the current default model:

```
miffi download -d path/to/download/dir
```

To download all available full models, use `--all` tag. A specific model can be downloaded by using `--model-name model_to_download`. Currently available models:

- `miffi_v1` - Model trained with finetuning and power spectrum input. (Default model)
- `miffi_no_ps_v1` - Model trained with finetuning but without power spectrum input. May be useful for filtering micrographs with pixel sizes larger than 1.5 Å.
- `miffi_no_pretrain_no_ps_v1` - Model trained without finetuning or power spectrum input. (not recommended for anything, for validation purpose only)

### Inference micrographs

Micrograph input can be provided as folder and wildcard to match (`--micdir micrograph_dir -w wildcard`), or can be extracted from a file (a star file, a cryosparc cs file, a plain text list, or a pkl file containing a list). Example for inferencing micrographs from a cs file:

```
miffi inference -f path/to/csfile -d path/to/data/dir --outname output_name \
                -g gpu_id_to_use -m path/to/model/dir
```

Details on arguments used above:

- `-f` Path to the file containing input micrograph list.
- `-d` Path to the data directory. This is often needed when micrograph path in the file is not absolute path. For star and cs files, this is usually the path to the root of the project directory.
- `--outname` Name to be prepended to the output result file.
- `-g` GPU id to use for inference.
- `-m` Path to the directory containing full model files.

Output containing prediction confidence scores will be saved as a dictionary in a pkl file.

### Catagorize micrographs

After inference, micrographs can be split into multiple catagories based on inference results. Output format can be one of the following: plain text list files, star files, or cs files. For outputing star and cs files, the original star file, or the original cs, passthrough cs, and csg files need to be provided. Example for catagorizing and outputing cs files:

```
miffi catagorize -t cs -i path/to/inference/result --cs path/to/original/csfile \
                 --ptcs path/to/original/passthrough/csfile --csg path/to/original/csgfile \
                 --sb --sc
```

Details on arguments used above:

- `-t` Type of the output, can be one of the following: `list`, `star`, `cs`.
- `-i` Path to the inference result file.
- `--cs` `--ptcs` `--csg` Path to the original cs, passthrough cs, and csg files, which are necessary for outputting cs files. If no passthrough cs file is available (e.g. import jobs), main cs file can be used for both cs and passthrough cs input. To output star files, path to the original star file needs to be provided through `--star`.
- `--sb` Split all singly bad micrographs into individual catagories. If this argument is not used, only film and minor crystalline catagories will be written.
- `--sc` Split all catagories into high and low confidence based on a cutoff value. Default confidence cutoff is 0.7, but can be configured through `--gc` for good predictions and `--bc` for bad predictions.

To load output cs files into cryosparc, use "Import Result Group" job and input the output csg file of the corresponding catagory.

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