#!/bin/bash

set -ex

source ~/.bashrc

#conda env create -f environment.yml
conda activate rnw
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
