import torch
from typing import Dict

from typing import Optional, List
from langchain.llms.base import LLM
from config.model_config import *
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

llm_model = LLM_MODEL
embedding_model = EMBEDDING_MODEL

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus
    device_map = {
        'transformer.word_embeddings': 0,
        'transformer.final_layernorm': 0,
        'lm_head': 0
    }
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


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    model_type: str = "chatglm"
    model_name_or_path: str = llm_model,
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.model_type == 'chatglm':
            response, _ = self.model.chat(self.tokenizer, prompt, history=self.history, max_length=self.max_token, temperature=self.temperature)
            torch_gc()
            if stop is not None:
                response = enforce_stop_tokens(response, stop)
            self.history = self.history + [[None, response]]
        return response

    def load_llm(self, llm_device=DEVICE, num_gpus='auto', device_map: Optional[Dict[str, int]] = None, **kwargs):
        if 'chatglm' in self.model_name_or_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained("llms/"+self.model_name_or_path, trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path))
            # if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            #     num_gpus = torch.cuda.device_count()
            #     if num_gpus < 2 and device_map is None:
            #         self.model = (AutoModel.from_pretrained(
            #             self.model_name_or_path, trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path),
            #             **kwargs).half().cuda())
            #     else:
            #         from accelerate import dispatch_model
            #         model = AutoModel.from_pretrained("llms/"+self.model_name_or_path,trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path), **kwargs).half()
            #         if device_map is None:
            #             device_map = auto_configure_device_map(num_gpus)
            #         self.model = dispatch_model(model, device_map=device_map)
            #         print("accelerating...")
            # else:
            #     self.model = (AutoModel.from_pretrained(
            #         self.model_name_or_path,
            #         trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path)).float().to(llm_device))
            self.model = (AutoModel.from_pretrained("llms/"+self.model_name_or_path, trust_remote_code=True, cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path), **kwargs).half().cuda())
            self.model = self.model.eval()
        else:
            print("loading error model name or path")
