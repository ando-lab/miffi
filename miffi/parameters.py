"""
Parameters for miffi
"""

DEFAULT_LABEL_NAMES = [['no_film','minor_film','major_film','film'],
                       ['no_crack_drift','crack_drift_empty'],
                       ['not_crystalline','minor_crystalline','major_crystalline'],
                       ['minor_contamination','major_contamination']]

AVAILABLE_MODELS = ['miffi_v1', 'miffi_no_ps_v1', 'miffi_no_pretrain_no_ps_v1']

CATEGORY_DEFAULT = ['good','bad_single','bad_film','bad_minor_crystalline','bad_multiple']

CATEGORY_ALL = ['good','bad_single','bad_film','bad_drift','bad_minor_crystalline',
                'bad_major_crystalline','bad_contamination','bad_multiple']

CATEGORY_GOOD_PREDICTIONS = [[0,1],[0],[0],[0]]

CONF_SPLIT_NAMES = ['high_conf','low_conf']

RECORD_URL = 'https://zenodo.org/api/records/8342009'

DEFAULT_DOWNLOAD = ['miffi_v1']
