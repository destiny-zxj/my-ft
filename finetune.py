"""

"""
import os
import yaml
import torch
import argparse
import subprocess
from trl import SFTTrainer
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TextStreamer

ALPACA_PROMPT = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""


class FtModelConfig:
    def __init__(self, args_filename="args.yaml"):
        if os.path.exists(args_filename):
            with open(args_filename, mode="r", encoding='utf-8') as fpr:
                args = yaml.load(fpr, Loader=yaml.FullLoader)
        else:
            args = dict()
        self.model_name = args['model_name'] if (
                    'model_name' in args and args['model_name'] is not None) else 'unsloth/llama-3-8b-bnb-4bit'
        #
        self.dataset_name = args['dataset'] if 'dataset' in args else None
        self.epochs = int(args['epochs']) if ('epochs' in args and args['epochs'] is not None) else 1
        self.max_steps = int(args['max_steps']) if ('max_steps' in args and args['max_steps'] is not None) else 60
        self.max_seq_length = int(args['max_seq_length']) if (
                    'max_seq_length' in args and args['max_seq_length'] is not None) else 4096
        self.logging_steps = int(args['logging_steps']) if ('logging_steps' in args and args['logging_steps'] is not None) else 1
        self.dtype = None
        self.load_in_4bit = bool(args['load_in_4bit']) if (
                    'load_in_4bit' in args and args['load_in_4bit'] is not None) else True
        self.alpaca_prompt = ALPACA_PROMPT
        #
        self.r = int(args['r']) if ('r' in args and args['r'] is not None) else 16
        self.target_modules = args['target_modules'] if (
                    'target_modules' in args and args['target_modules'] is not None) else ["q_proj", "k_proj", "v_proj",
                                                                                           "o_proj", "gate_proj",
                                                                                           "up_proj", "down_proj"]
        self.lora_alpha = int(args['lora_alpha']) if ('lora_alpha' in args and args['lora_alpha'] is not None) else 16
        # lora
        self.lora_output_dir = args['lora_output_dir'] if (
                    'lora_output_dir' in args and args['lora_output_dir'] is not None) else 'lora_model'
        self.lora_save_as_gguf = bool(args['lora_save_as_gguf']) if (
                    'lora_save_as_gguf' in args and args['lora_save_as_gguf'] is not None) else False
        self.lora_save_method = args['lora_save_method'] if (
                    'lora_save_method' in args and args['lora_save_method'] is not None) else 'merged_16bit'
        self.lora_quantization_method = args['lora_quantization_method'] if (
                    'lora_quantization_method' in args and args['lora_quantization_method'] is not None) else 'q4_0'
        # save
        self.save_model_name = args['save_model_name'] if (
                    'save_model_name' in args and args['save_model_name'] is not None) else 'my_model'
        self.quantization_type = args['quantization_type'] if (
                    'quantization_type' in args and args['quantization_type'] is not None) else 'Q8_0'


class FtUtil:
    def __init__(self):
        self.start_gpu_memory = None
        self.max_memory = None

    def show_status(self):
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        self.start_gpu_memory = start_gpu_memory
        self.max_memory = max_memory

    def show_final(self, trainer_stats):
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - self.start_gpu_memory, 3)
        used_percentage = round(used_memory / self.max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / self.max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    @staticmethod
    def save_gguf(config: FtModelConfig):
        """
        保存 gguf 格式模型
        :param config:
        :return:
        """
        assert os.path.exists(config.model_name), f"[>_<] Model not found at {config.model_name}"
        # model -> gguf
        output_filename = f"{config.save_model_name}.gguf"
        convert_process = subprocess.Popen(
            [
                "./llama.cpp/convert-hf-to-gguf.py",
                "--outfile", output_filename,
                "--outtype", "f16",
                config.model_name
            ]
        )
        convert_process.wait()
        print(f"[^_^] Save to {output_filename}")

    @staticmethod
    def quantize(config: FtModelConfig):
        """
        量化模型
        :return:
        """
        model_name = f"{config.save_model_name}.gguf"
        assert os.path.exists(model_name), f"[>_<] Model not found: {model_name}"
        # 量化 gguf
        output_filename = f"{config.save_model_name}.{config.quantization_type}.gguf"
        quantize_process = subprocess.Popen(
            [
                "./llama.cpp/quantize",
                model_name,
                output_filename,
                config.quantization_type
            ]
        )
        quantize_process.wait()
        print(f"[^_^] Save to {output_filename}")


class FtDataset:
    def __init__(self, data_filename: str, tokenizer, alpaca_prompt: str, batched=True):
        self.data_filename = data_filename
        self.EOS_TOKEN = tokenizer.eos_token
        self.alpaca_prompt = alpaca_prompt
        self.batched = batched
        self.dataset = self.load_dataset()

    def load_dataset(self):
        if os.path.exists(self.data_filename):
            data_dict = {
                "train": self.data_filename
            }
            dataset = load_dataset("json", data_files=data_dict, split="train")
        else:
            dataset = load_dataset(self.data_filename, split="train")
        dataset = dataset.map(self.formatting_prompts_func, batched=self.batched)
        return dataset

    def formatting_prompts_func(self, examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for item_instruction, item_input, item_output in zip(instructions, inputs, outputs):
            text = self.alpaca_prompt.format(item_instruction, item_input, item_output) + self.EOS_TOKEN
            texts.append(text)
        return {"text": texts, }


class FtModel:
    def __init__(self, config: FtModelConfig):
        self.config = config
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit
        )

    def train(self):
        """
        加载训练模式
        :return:
        """
        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            r=self.config.r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        self.model.print_trainable_parameters()


class FtTrainer:
    def __init__(
            self, ft_model: FtModel, ft_dataset: FtDataset, config: FtModelConfig
    ):
        training_args = TrainingArguments(
            num_train_epochs=config.epochs,
            max_steps=config.max_steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        )
        self.trainer = SFTTrainer(
            model=ft_model.model,
            tokenizer=ft_model.tokenizer,
            train_dataset=ft_dataset.dataset,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=training_args,
        )


class FtInference:
    def __init__(self, ft_model: FtModel, config: FtModelConfig):
        self.model = ft_model.model
        self.tokenizer = ft_model.tokenizer
        self.config = config

    def inference(self, instruction: str, input_text="", output_text=""):
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference
        inputs = self.tokenizer(
            [
                ALPACA_PROMPT.format(
                    instruction,  # instruction
                    input_text,  # input
                    output_text,  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")
        text_streamer = TextStreamer(self.tokenizer)
        _ = self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


class FtSave:
    def __init__(self, model: FtModel, config: FtModelConfig):
        """

        :param model:
        """
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.config = config

    def save_lora(self):
        """
        保存 lora 模型
        :return:
        """
        self.model.save_pretrained(self.config.lora_output_dir)
        self.tokenizer.save_pretrained(self.config.lora_output_dir)
        print(f"[^_^] Lora model save to {self.config.lora_output_dir}")

    def save_model(self):
        """
        合并模型
        :return:
        """
        if self.config.lora_save_as_gguf:
            self.model.save_pretrained_gguf(
                self.config.save_model_name, self.tokenizer, quantization_method=self.config.lora_quantization_method
            )
        else:
            self.model.save_pretrained_merged(
                self.config.save_model_name, self.tokenizer, save_method=self.config.lora_save_method
            )
        print(f"[^_^] Model save to {self.config.save_model_name}")


def train(args):
    print(vars(args))
    ft_config = FtModelConfig()
    if args.epochs is not None:
        ft_config.epochs = args.epochs
    if args.data is not None:
        ft_config.dataset_name = args.data
    if args.lora_dir is not None:
        ft_config.lora_output_dir = args.lora_dir
    if args.model_name is not None:
        ft_config.model_name = args.model_name
    print(vars(ft_config))
    ft_model = FtModel(ft_config)
    ft_dataset = FtDataset(
        data_filename=ft_config.dataset_name,
        tokenizer=ft_model.tokenizer,
        alpaca_prompt=ft_config.alpaca_prompt
    )
    util = FtUtil()
    util.show_status()
    ft_model.train()
    ft_trainer = FtTrainer(ft_model, ft_dataset, ft_config)
    trainer_stats = ft_trainer.trainer.train()
    util.show_final(trainer_stats)
    ft_save = FtSave(ft_model, ft_config)
    ft_save.save_lora()


def inference(args):
    ft_config = FtModelConfig()
    ft_config.model_name = args.model_name
    ft_model = FtModel(ft_config)
    ft_inference = FtInference(ft_model, ft_config)
    while True:
        instruction = input(r"请输入指令(退出`\exit`): ")  # "只用中文回答问题"
        if instruction == r"\exit":
            break
        input_text = input("请输入文本(可选): ")  # "海绵宝宝是不是海绵体？"
        ft_inference.inference(
            instruction=instruction,
            input_text=input_text
        )


def save(args):
    print(vars(args))
    ft_config = FtModelConfig()
    ft_config.model_name = ft_config.lora_output_dir
    if args.lora_dir is not None:
        ft_config.lora_output_dir = args.lora_dir
    if args.model_name is not None:
        ft_config.save_model_name = args.model_name
    if args.gguf is not None:
        ft_config.lora_save_as_gguf = True
    print(vars(ft_config))
    ft_model = FtModel(ft_config)
    ft_save = FtSave(ft_model, ft_config)
    ft_save.save_model()


def download(args):
    print(vars(args))
    ft_config = FtModelConfig()
    if args.model_name is not None:
        ft_config.model_name = args.model_name
    if args.output is not None:
        ft_config.save_model_name = args.output
    print(vars(ft_config))
    ft_model = FtModel(ft_config)
    ft_model.model.save_pretrained(ft_config.save_model_name)
    ft_model.tokenizer.save_pretrained(ft_config.save_model_name)
    print(f"[^_^] 下载完成!")


def convert(args):
    ft_config = FtModelConfig()
    ft_config.model_name = args.model_name
    if args.output is not None:
        ft_config.save_model_name = args.output
    FtUtil.save_gguf(ft_config)
    if args.quantize:
        FtUtil.quantize(ft_config)


def quantize(args):
    ft_config = FtModelConfig()
    if args.model_name is not None:
        ft_config.save_model_name = args.model_name
    FtUtil.quantize(ft_config)


def main():
    # parser
    parser = argparse.ArgumentParser(description="LLM finetune tool.")
    subparsers = parser.add_subparsers(dest="mode")
    # train
    train_parser = subparsers.add_parser(name="train", help="Train mode.")
    train_parser.add_argument('-e', '--epochs', type=int, required=False)
    train_parser.add_argument('-m', '--model_name', type=str, required=False)
    train_parser.add_argument('-d', '--data', type=str, required=False)
    train_parser.add_argument('--lora_dir', type=str, required=False)
    train_parser.set_defaults(func=train)
    # inference
    inference_parser = subparsers.add_parser(name="inference", help="推理模式.")
    inference_parser.add_argument('-m', '--model_name', type=str, required=True, help='模型名称')
    inference_parser.set_defaults(func=inference)
    # save
    save_parser = subparsers.add_parser(name="save", help="保存模式(to `hf`)")
    save_parser.add_argument('--lora_dir', type=str, required=False, help='Lora 模型文件夹.')
    save_parser.add_argument('-m', '--model_name', type=str, required=False, help='输出模型文件夹.')
    save_parser.add_argument('--gguf', type=bool, required=False, help='输出 GGUF 格式.')
    save_parser.set_defaults(func=save)
    # download
    download_parser = subparsers.add_parser(name="download", help="下载并保存模型.")
    download_parser.add_argument('-m', '--model_name', type=str, required=False)
    download_parser.add_argument('-o', '--output', type=str, required=False)
    download_parser.set_defaults(func=download)
    # convert
    convert_parser = subparsers.add_parser(name="convert", help="转换模式")
    convert_parser.add_argument('-m', '--model_name', type=str, required=True, help='模型路径或名称.')
    convert_parser.add_argument('-o', '--output', type=str, required=False, help='模型输出名称.')
    convert_parser.add_argument('-q', '--quantize', type=bool, default=True, help='是否量化.')
    convert_parser.set_defaults(func=convert)
    # quantize
    quantize_parser = subparsers.add_parser(name='quantize', help="量化模式.")
    quantize_parser.add_argument('-m', '--model_name', type=str, required=False)
    quantize_parser.set_defaults(func=quantize)
    #
    args = parser.parse_args()
    if 'func' in args:
        args.func(args)


if __name__ == '__main__':
    main()
