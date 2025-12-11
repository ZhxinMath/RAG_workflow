import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# 修改导入路径
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
import torch

# --- 步骤 1: 离线下载模型 (如果尚未下载) ---
# 你可以在 AutoDL 的终端中手动运行以下 Python 代码来下载模型到本地缓存
# 这样可以避免在训练/向量化时因网络问题而失败
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"正在加载模型 {model_name} 的 Tokenizer 和 Model 以触发下载...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("模型已成功加载到内存并缓存到本地。")
except Exception as e:
    print(f"尝试加载模型时出错 (这可能意味着需要先离线下载): {e}")
    print("请确保网络连接，或尝试使用 AutoDL 的镜像下载功能，或者手动下载模型。")
    print("如果下载成功，后续运行将从缓存加载。")
    # 即使这里出错，也继续尝试初始化 embeddings，因为它可能会自己处理缓存
    pass

# --- 步骤 2: 初始化 Embeddings ---
# HuggingFaceEmbeddings 会自动从本地缓存加载模型
print("正在初始化 HuggingFaceEmbeddings...")
# 可以尝试添加一些参数来提高网络请求的鲁棒性，但主要问题还是网络
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"}, # 如果 GPU 内存不够，可以强制使用 CPU
    encode_kwargs={"normalize_embeddings": True}, # 一些编码参数
)

# --- 步骤 3: 加载、分块 PDF ---
pdf_path = "/mnt/sda/home/XuYH/daqing_rag/data/Tesla_Manual.pdf"
print(f"正在加载 PDF: {pdf_path} ...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"PDF 加载完成，共 {len(documents)} 页。")

print("正在对文档进行分块...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个块的最大字符数
    chunk_overlap=50,      # 相邻块之间的重叠字符数
)
chunks = text_splitter.split_documents(documents)
print(f"分块完成，共得到 {len(chunks)} 个文本块。")

# --- 步骤 4: 向量化并保存 ---
print("开始向量化过程...")
vectorstore = FAISS.from_documents(chunks, embeddings)

save_path = "/mnt/sda/home/XuYH/daqing_rag/vectorstore/faiss_tesla_manual"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

print(f"正在将向量数据库保存到 {save_path} ...")
vectorstore.save_local(save_path)
print("向量数据库已成功保存！")

# --- 下一步使用示例 ---
# 你可以用以下代码加载并检索：
# loaded_vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
# query = "关于充电接口的信息"
# docs = loaded_vectorstore.similarity_search(query, k=3)
# for doc in docs:
#     print(doc.page_content)
#     print("---")
