#!/bin/bash

# Make sure the config has the weights that you would like to test
python train.py configs/debug.yaml --test_only --dump_rpn rpn_out.csv --outdir rpn_example

# See rpn_analysis for more options and documentation
python rpn_analysis.py --type recall --rpn_file ./rpn_example/output_104512/rpn_out.csv
