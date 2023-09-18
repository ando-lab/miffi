import logging

try:
    from miffi._version import version as __version__
except ModuleNotFoundError:
    __version__ = 'src'

logging.basicConfig(format='(%(levelname)s|%(filename)s|%(asctime)s) %(message)s', level=logging.INFO, 
                    datefmt='%d-%b-%y %H:%M:%S')
