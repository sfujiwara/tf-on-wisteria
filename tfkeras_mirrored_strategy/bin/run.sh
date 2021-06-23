#!/bin/bash

#PJM -g gt00
#PJM -L rscgrp=tutorial-share
#PJM -L gpu=4
#PJM -L elapse=1:00:00

module purge
module load cuda/11.2
module load cudnn/8.1.0
module list

source tf/bin/activate

python main.py

