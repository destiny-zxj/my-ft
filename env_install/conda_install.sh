#!/bin/sh

#conda create --name ft_env python=3.10
#conda init
#conda activate ft_env

conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

proxychains4 pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

proxychains4 pip install --no-deps trl peft accelerate bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
