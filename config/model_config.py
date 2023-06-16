import torch
import logging, uuid, os


LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "text2vec-base": "GanymedeNil/text2vec-base-chinese",
    'simbert-base-chinese': 'WangZeJun/simbert-base-chinese',
    'paraphrase-multilingual-MiniLM-L12-v2': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

llm_model_dict = {
    "chatglm": {
        "ChatGLM-6B": "THUDM/chatglm-6b",
        "ChatGLM-6B-int4": "THUDM/chatglm-6b-int4",
        "ChatGLM-6B-int8": "THUDM/chatglm-6b-int8",
        "ChatGLM-6b-int4-qe": "THUDM/chatglm-6b-int4-qe"
    },
    "belle": {
        "BELLE-LLaMA": "belle",
    },
    "vicuna": {
        "Vicuna-LLaMA": "vicuna",
    },
    "bgi-med-chatglm-6b": {
        "bgi-med-chatglm-6b": "bgi-med-chatglm-6b",
    },
}

# LLM model name
LLM_MODEL = "bgi-med-chatglm-6b"
EMBEDDING_MODEL = "text2vec-base"
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')

# LLM streaming reponse
STREAMING = False

# LLM embeding
IS_EMBEDDING = False
USE_WEB = False
VS_PATH = "/data/vs_data/"
VS_INDEX = "faiss_index"

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

# LLM input history length
LLM_HISTORY_LEN = 3
# 文本分句长度
SENTENCE_SIZE = 100

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
