from datetime import datetime

DATE_FORMAT = '%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)


WARMUP_TIME = 20
RUNNING_TIME = 20

# hyperparameter
# BATCH_SIZE = 1
# HEAD_NUM = 2
# SEQ_LEN = 2
# HEAD_DIM = 8

BATCH_SIZE = 1
HEAD_NUM = 16
SEQ_LEN = 1024
HEAD_DIM = 32

MASK_ID = 1 # 

LAYER_NUM = 1
AVG_SEQ_LEN = -1
DATA_TYPE = "fp16"
