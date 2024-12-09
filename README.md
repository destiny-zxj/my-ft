# MyFinetune

## 注意

* 禁止在 Windows 平台下运行！
* 显卡需求至少 8GB (某些情况下 WSL 8G 爆显存)
* 因为需要访问 huggingface，某些情况下需要`科学上网`

## 安装 (与 unsloth 一致)

### conda

```shell
conda create --name ft_env python=3.10
conda activate ft_env

conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps trl peft accelerate bitsandbytes
```

### pip

```shell
# 建议使用 conda
```

### llama.cpp

```shell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make LLAMA_CUDA=1 -j8
```

## 使用

```shell
# 使用以下命令查看使用方法
python finetune.py -h
```
