import gc
import time
from pathlib import Path
from config.model_config import *
import torch
from typing import Dict
import transformers
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer)

class LoadModel:
    """加载自定义模型"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device_map = None
        self.model_name = LLM_MODEL

    def load_model(self):
        """加载自定义路径模型"""
        print(f"Loading {self.model_name}...")
        t0 = time.time()

        model_path = Path(str(Path.cwd())+f'/model/{self.model_name}')

        if 'chatglm' in self.model_name.lower():
            LoaderClass = AutoModel
        else:
            LoaderClass = AutoModelForCausalLM

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2:
                model = LoaderClass.from_pretrained(model_path, trust_remote_code=True).half().cuda()
            else:
                from accelerate import dispatch_model
                self.device_map = self.chatglm_conf_device_map(num_gpus)
                model = LoaderClass.from_pretrained(model_path, trust_remote_code=True).half()
                model = dispatch_model(model, device_map=self.device_map)
        else:
            print("no GPU is running!!!")

        # Loading the tokenizer
        if type(model) is transformers.LlamaForCausalLM:
            tokenizer = LlamaTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
            try:
                tokenizer.eos_token_id = 2
                tokenizer.bos_token_id = 1
                tokenizer.pad_token_id = 0
            except Exception as e:
                print(e)
                pass
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"Loaded the model in {(time.time() - t0):.2f} sedconds.")
        return model, tokenizer

    def torch_gc(self):
        gc.collect()
        if torch.has_cuda:
            device_id = "0" if torch.cuda.is_available() else None
            CUDA_DEVICE = f"cuda:{device_id}" if device_id else "cuda"
            with torch.cuda.device(CUDA_DEVICE):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        elif torch.has_mps:
            try:
                from torch.mps import empty_cache
                empty_cache()
            except Exception as e:
                print(e)
                print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")
        else:
            print("未检测到其他cuda或mps，暂不支持清理显存")

    def chatglm_conf_device_map(self, num_gpus: int) -> Dict[str, int]:
        num_trans_layers = 28
        per_gpu_layers = 30 / num_gpus
        device_map = {f'transformer.word_embeddings': 0,
                      f'transformer.final_layernorm': 0, 'lm_head': 0, }

        used = 2
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f'transformer.layers.{i}'] = gpu_target
            used += 1
        return device_map

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = self.tokenizer = None
        self.torch_gc()

    def reload_model(self):
        self.unload_model()
        self.model, self.tokenizer = self.load_model(self.modle_name)
        self.model = self.model.eval()