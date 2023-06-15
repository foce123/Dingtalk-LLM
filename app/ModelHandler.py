import sentence_transformers
import torch
from duckduckgo_search import DDGS
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.docstore.document import Document

from models.chatglm_llm import ChatGLM
from config.model_config import *

embedding_model_dict = embedding_model_dict
llm_model_dict = llm_model_dict
EMBEDDING_DEVICE = EMBEDDING_DEVICE
LLM_DEVICE = LLM_DEVICE
num_gpus = torch.cuda.device_count()
llm_model = LLM_MODEL
embedding_model = EMBEDDING_MODEL
is_embedding = IS_EMBEDDING
vs_index = VS_INDEX
use_web = USE_WEB

llm_model_list = []


def search_web(query):
    with DDGS(proxies="socks5://localhost:9150", timeout=20) as ddgs:
        results = ddgs.answers(query)
    web_content = ''
    if results:
        for result in results:
            web_content += result['body']
    return web_content


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


class ModelQALLM:
    llm: object = None
    embeddings: object = None
    is_embedding: bool = False
    history_len: int = 3
    top_k: int = 6

    def init_model_config(self, isembedding: bool = False):
        if isembedding:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model])
            self.embeddings.client = sentence_transformers.SentenceTransformer(
                self.embeddings.model_name,
                device=EMBEDDING_DEVICE,
                cache_folder=os.path.join(MODEL_CACHE_PATH, self.embeddings.model_name))
        self.llm = ChatGLM()
        if 'chatglm' == llm_model.lower():
            self.llm.model_type = 'chatglm'
            self.llm.model_name_or_path = llm_model_dict['chatglm'][llm_model]
        # elif 'belle' in llm_model.lower():
        #     self.llm.model_type = 'belle'
        #     self.llm.model_name_or_path = llm_model_dict['belle'][llm_model]
        # elif 'vicuna' in llm_model.lower():
        #     self.llm.model_type = 'vicuna'
        #     self.llm.model_name_or_path = llm_model_dict['vicuna'][llm_model]
        elif 'bgi-med-chatglm-6b' == llm_model.lower():
            self.llm.model_type = 'chatglm'
            self.llm.model_name_or_path = llm_model_dict['bgi-med-chatglm-6b'][llm_model]
        self.llm.load_llm(llm_device=LLM_DEVICE, num_gpus=num_gpus)

    def get_llm_answer(self, query, web_content, top_k: int = 6, history_len: int = 3, temperature: float = 0.01, top_p: float = 0.1, history=[]):
        self.llm.temperature = temperature
        self.llm.top_p = top_p
        self.history_len = history_len
        self.top_k = top_k
        if web_content:
            prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                已知网络检索内容：{web_content}""" + """
                                已知内容:
                                {context}
                                问题:
                                {question}"""
        else:
            prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。
                已知内容:
                {context}
                问题:
                {question}"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.llm.history = history[-self.history_len:] if self.history_len > 0 else []
        if is_embedding:
            # select your vector store to load
            vector_store = FAISS.load_local(vs_index, self.embeddings)
            answer_chain = RetrievalQA.from_llm(llm=self.llm, retriever=vector_store.as_retriever(search_kwargs={"k": self.top_k}), prompt=prompt)
            answer_chain.combine_documents_chain.document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
            # answer_chain.return_source_documents = True
            result = answer_chain({"query": query})
        else:
            from langchain import LLMChain
            answer_chain = LLMChain(prompt=prompt, llm=self.llm)
            result = answer_chain.predict(context="", question=query)
            print(result)
        return result


model_chat_llm = ModelQALLM()


def init_model():
    try:
        model_chat_llm.init_model_config()
        model_chat_llm.llm._call("你好")
        return """初始模型已成功加载，可以开始对话"""
    except Exception as e:
        print(e)
        return """模型未成功加载，请重新选择模型后点击"重新加载模型"按钮"""


def clear_session():
    return '', None


def reinit_model(is_embedding=is_embedding):
    try:
        model_chat_llm.init_model_config(isembedding=is_embedding)
        model_status = """模型已成功重新加载，可以开始对话"""
    except Exception as e:
        model_status = """模型未成功重新加载，请点击重新加载模型"""
    return [[None, model_status]]


def predict(input, use_web: bool = False, top_k: int = 6, history_len: int = 3, temperature: float = 0.01, top_p: float = 0.1, history=None):
    if history == None:
        history = []
    if use_web:
        from agent.bing_search import bing_search
        results = bing_search(input)
        result_docs = search_result2docs(results)
        web_content = "\n".join([doc.page_content for doc in result_docs])
        # web_content = search_web(query=input)
    else:
        web_content = ''
    resp = model_chat_llm.get_llm_answer(
        query=input,
        web_content=web_content,
        top_k=top_k,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        history=history)
    #history.append((input, resp['result']))
    print(type(resp))
    history.append((input, resp))
    return resp, history

# model_status = init_model()
# print(model_status)
