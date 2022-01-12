#!/bin/bash

# run from parent directory

python train.py configs/first_10.yaml --outdir voc_10_10
python train.py configs/last_10.yaml --outdir voc_10_10 --load_last