#!/bin/bash

#PJM -g gt00
#PJM -L rscgrp=tutorial-share
#PJM -L gpu=4
#PJM -L elapse=1:00:00
#PJM -o logs/stdout_gpu4.log
#PJM -e logs/stderr_gpu4.log


module purge
module load cuda/11.2
module load cudnn/8.1.0
module load nccl/2.9.6
module list

source ../.venv/bin/activate

python main.py

