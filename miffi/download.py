"""
Download miffi model files
"""

import logging
import sys
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib
import json
import time
import datetime
from .parameters import AVAILABLE_MODELS, RECORD_URL, DEFAULT_DOWNLOAD

logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument(
        '-d',
        '--download-dir',
        type=Path,
        default=Path.cwd(),
        help="Path to the directory for downloading model files"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Download all available miffi models",
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help=f"Specify the name of the model to download. Available options: {', '.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="Overwrite existing files that have mismatched hash",
    )

def get_filehash(filepath, algorithm):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.new(algorithm)
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def main(args):
    if args.all:
        models_to_download = AVAILABLE_MODELS
    elif args.model_name is not None:
        assert args.model_name.lower() in AVAILABLE_MODELS, f"Specified model name {args.model_name} is not available!"
        models_to_download = [args.model_name.lower()]
    else:
        models_to_download = DEFAULT_DOWNLOAD
    logger.info(f"Download the following models: {', '.join(models_to_download)}")
    
    r = requests.get(RECORD_URL)
    if r.ok:
        files = json.loads(r.text)['files']
        for file in files:
            if file['key'].split('.')[0] in models_to_download:
                logger.info(f"Start downloading file {file['key']}")
                download_path = args.download_dir / file['key']
                checksum = file['checksum'].split(':')
                
                if download_path.is_file():
                    logger.info(f"File {file['key']} already exsits in download folder, checking hash")
                    file_hash = get_filehash(download_path, checksum[0])
                    if file_hash == checksum[1]:
                        logger.info(f"Matched hash, skip downloading {file['key']}")
                        continue
                    elif args.overwrite:
                        logger.info(f"Mismatched hash! Overwriting exisitng file")
                    else:
                        raise Exception(f"Mismatched hash for existing file {file['key']}! If you want to overwrite the file, use --overwrite flag")
                
                file_url = file['links']['self']
                file_size = file['size']
                chunk_size = 4096
                for try_idx in range(4):
                    try:
                        file_req = requests.get(file_url, stream=True)
                        if file_req.ok:
                            with open(download_path, 'wb') as fd:
                                with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
                                    for idx, chunk in enumerate(file_req.iter_content(chunk_size=chunk_size)):
                                        fd.write(chunk)
                                        pbar.update(min(chunk_size,file_size-idx*chunk_size))
                            logger.info(f"Finished downloading {file['key']}")
                            logger.info(f"Checking hash of the downloaded file")
                            file_hash = get_filehash(download_path, checksum[0])
                            if file_hash == checksum[1]:
                                logger.info(f"Matched hash, download was successful")
                            else:
                                raise Exception(f"Mismatched hash from downloaded file! Attempt to download again ...")
                        else:
                            raise Exception(f"Unable to request {file['key']} from server with status code {file_req.status_code}")
                    except Exception as e:
                        if try_idx < 3:
                            wait_time = 10
                            logger.error(f"The following error occured: {e}")
                            logger.error(f"Retry after {wait_time} seconds")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to download {file['key']} after multiple tries")
                    else:
                        break    
    else:
        raise Exception(f"Failed to access record on server with status code {r.status_code}! Code version too old?")

    logger.info("All done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())