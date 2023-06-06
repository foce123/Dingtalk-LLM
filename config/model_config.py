import torch.cuda
import torch.backends
import logging, uuid, os


LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

llm_model_dict = {
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
    "bgi-med-chatglm-6b": {
        "name": "bgi-med-chatglm-6b",
        "pretrained_model_name": "bgi-med-chatglm-6b",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
}

# LLM model name
LLM_MODEL = "bgi-med-chatglm-6b"

# LLM streaming reponse
STREAMING = True

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

# LLM input history length
LLM_HISTORY_LEN = 3

FLAG_USER_NAME = uuid.uuid4().hex

logger.info(f"""
loading model config
llm device: {LLM_DEVICE}
dir: {os.path.dirname(os.path.dirname(__file__))}
flagging username: {FLAG_USER_NAME}
""")

# Bing 搜索必备变量
# 使用 Bing 搜索需要使用 Bing Subscription Key
# 具体申请方式请见 https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
BING_SUBSCRIPTION_KEY = ""
