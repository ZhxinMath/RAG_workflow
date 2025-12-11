import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. 重新初始化 Embeddings (必须与保存时使用的配置一致) ---
# 由于之前是用 CPU 保存的，加载时也需要配置为 CPU
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # 与保存时一致
    encode_kwargs={"normalize_embeddings": True},
)

# --- 2. 加载向量数据库 ---
# 注意：使用你实际保存的路径，这里假设是 CPU 版本的路径
vectorstore_path = "/mnt/sda/home/XuYH/daqing_rag/vectorstore/faiss_tesla_manual"

print(f"正在从 {vectorstore_path} 加载向量数据库...")
# allow_dangerous_deserialization=True 是因为 FAISS 保存了自定义对象 (pickle)
vectorstore = FAISS.load_local(
    vectorstore_path, 
    embeddings, 
    allow_dangerous_deserialization=True
)
print("向量数据库加载完成！")

# --- 3. 定义检索函数 ---
def retrieve_relevant_chunks(question: str, k: int = 3):
    """
    根据问题检索最相关的文本块。

    Args:
        question (str): 输入的问题。
        k (int): 检索返回的最相关块的数量。 默认为 3。

    Returns:
        List[str]: 检索到的最相关文本块列表。
    """
    print(f"\n正在检索与问题 '{question}' 相关的文本块 (top-{k})...")
    # similarity_search 是 LangChain 提供的 top-k 检索方法
    docs = vectorstore.similarity_search(question, k=k)
    
    # 提取检索到的文档内容
    retrieved_chunks = [doc.page_content for doc in docs]
    
    print(f"找到 {len(retrieved_chunks)} 个相关文本块。")
    return retrieved_chunks

# --- 4. 示例使用 ---
if __name__ == "__main__":
    # 你可以修改这里的 question 和 k 值来测试
    # user_question = "如何为 Tesla 汽车充电？"
    # top_k = 3

    # # 执行检索
    # relevant_chunks = retrieve_relevant_chunks(user_question, top_k)

    # # 打印结果
    # print(f"\n--- 与问题 '{user_question}' 最相关的 {top_k} 个文本块 ---")
    # for i, chunk in enumerate(relevant_chunks):
    #     print(f"\n--- 相关块 {i+1} ---")
    #     # 可能需要处理一下换行符，使其更易读
    #     clean_chunk = chunk.replace('\n', ' ').strip() 
    #     print(clean_chunk[:500] + "..." if len(clean_chunk) > 500 else clean_chunk) # 限制打印长度
    #     print("-" * 20)

    # 也可以交互式输入
    print("\n--- 交互式检索 ---")
    while True:
        question = input("\n请输入你的问题 (输入 'quit' 退出): ")
        if question.lower() == 'quit':
            break
        k_input = input("请输入 top-k 的 k 值 (默认为 3): ")
        try:
            k_val = int(k_input) if k_input.strip() else 3
        except ValueError:
            k_val = 3
        chunks = retrieve_relevant_chunks(question, k_val)
        print(f"\n--- 与问题 '{question}' 最相关的 {k_val} 个文本块 ---")
        for i, chunk in enumerate(chunks):
            print(f"\n--- 相关块 {i+1} ---")
            clean_chunk = chunk.replace('\n', ' ').strip()
            print(clean_chunk[:500] + "..." if len(clean_chunk) > 500 else clean_chunk)
            print("-" * 20)