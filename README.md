# TensorFlow on Wisteria

## Connect to login node

```
ssh -l ${USER_ID} wisteria.cc.u-tokyo.ac.jp
```

## Setup Python

### Connect to computation node

```
pjsub --interact -g gt00 -L rscgrp=tut2-interactive-a,node=1
```

```
module purge
module load cuda/11.2
module load cudnn/8.1.0
```

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install tensorflow
```
