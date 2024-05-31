#!/bin/bash
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-12-4p
# sudo apt-get install git-lfs -y
# git lfs install
export PYTHONPATH=$PWD:$PYTHONPATH
pip install -r requirements.txt
cd scripts && bash prepare_eval_data.sh
cd scripts
cd indic_eval
git clone https://huggingface.co/datasets/Divyanshu/indicxnli
export HF_HOME=$PWD
export TRANSFORMERS_CACHE=$PWD
# bash ./indicxnli.sh
#pip install timm
#time python3 download_model.py 'hf_hub:timm/vgg16.tv_in1k'
#scripts/prepare_eval_data.sh
time scripts/eval/codex_humaneval.sh
