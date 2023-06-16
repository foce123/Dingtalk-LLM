# coding=utf8
import os
import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
# from chinese_text_splitter import ChineseTextSplitter
from langchain.vectorstores import FAISS
from config.model_config import embedding_model_dict, EMBEDDING_MODEL, VS_INDEX, VS_PATH, EMBEDDING_DEVICE, MODEL_CACHE_PATH, SENTENCE_SIZE
from utils.textspliter import ChineseTextSplitter

class VsInit:
    def __init__(self):
        self.filepath = VS_PATH
        self.index = VS_INDEX
        self.sentence_size = SENTENCE_SIZE
        self.embedding_model_dict = embedding_model_dict
        self.embedding_model = EMBEDDING_MODEL
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[self.embedding_model])
        self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name,
            device=EMBEDDING_DEVICE,
            cache_folder=os.path.join(MODEL_CACHE_PATH, self.embeddings.model_name))

    def load_file(self):
        if self.filepath.lower().endswith(".md"):
            loader = UnstructuredFileLoader(self.filepath, mode="elements")
            docs = loader.load()
        elif self.filepath.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(self.filepath)
            textsplitter = ChineseTextSplitter(pdf=True, sentence_size=self.sentence_size)
            docs = loader.load_and_split(textsplitter)
        else:
            loader = UnstructuredFileLoader(self.filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=self.sentence_size)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs

    def init_vector_store(self):
        docs = self.load_file()
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local(self.index)
        # return vector_store
