from abc import ABC
from utils.model_load import LoadModel
from typing import Optional, List
from langchain.llms.base import LLM


class ChatGLM(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoadModel = None
    history_len: int = 10

    def __init__(self, checkPoint: LoadModel = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str) -> str:
        response, _ = self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature
        )
        return response

    def getAnswer(self, prompt: str, history: List[List[str]] = []):
        response, _ =self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            prompt,
            history=history[-self.history_len:] if self.history_len > 0 else[],
            max_lenth=self.max_token,
            temperature=self.temperature
        )
        self.checkPoint.torch_gc()
        history += [[prompt, response]]
        return response
