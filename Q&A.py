import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. 配置 API Key 和模型 ---
os.environ["OPENAI_API_KEY"] = "sk-9750b158612748ceb66459b5346b9291"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  

llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base=DEEPSEEK_BASE_URL,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
)

# --- 2. 加载向量数据库 ---
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore_path = "daqing_rag/RAG_workflow/vectorstore/faiss_tesla_manual"
print(f"正在从 {vectorstore_path} 加载向量数据库...")
retriever = FAISS.load_local(
    vectorstore_path, 
    embeddings, 
    allow_dangerous_deserialization=True
).as_retriever(search_kwargs={"k": 3})
print("向量数据库加载完成！")

# --- 3. 定义 Prompt 模板 ---
template = """
你是一个知识渊博的助手。请仅基于以下提供的上下文信息来回答用户的问题。
如果你无法从上下文信息中找到答案，请直接回答“根据提供的上下文，无法回答该问题。”
问题: {question}
上下文: {context}
回答:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 4. 定义格式化函数 ---
def format_docs(docs):
    """将检索到的文档列表格式化为一个字符串"""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# --- 5. 定义修正后的查询函数（不再依赖 RunnableParallel）---
def query_rag_with_context(question: str):
    """
    执行 RAG 查询，并返回模型回答和检索到的上下文。

    Args:
        question (str): 输入的问题。

    Returns:
        dict: 包含 'response' (模型回答) 和 'context' (检索到的 Document 列表) 的字典。
    """
    print(f"正在处理问题: '{question}'")
    try:
        # Step 1: 检索相关文档
        docs = retriever.invoke(question)
        
        if not docs:
            print("警告：未检索到任何相关文档。")
            return {
                "response": "根据提供的上下文，无法回答该问题。",
                "context": []
            }

        # Step 2: 格式化上下文
        formatted_context = format_docs(docs)

        # Step 3: 构造完整提示并调用 LLM
        messages = prompt.format_messages(question=question, context=formatted_context)
        response = llm.invoke(messages)
        answer = response.content

        return {
            "response": answer,
            "context": docs  # 保留原始 Document 对象用于展示
        }

    except Exception as e:
        print(f"查询过程中发生错误: {e}")
        return {
            "response": "抱歉，查询过程中发生了错误。",
            "context": []
        }

# --- 6. 示例使用 ---
if __name__ == "__main__":
    user_question = "轮胎坏了怎么办？"
    
    result = query_rag_with_context(user_question)
    
    print(f"\n--- 问题 ---\n{user_question}\n")
    
    print("--- 检索到的相关文本块 ---")
    if result["context"]:
        for i, doc in enumerate(result["context"]):
            print(f"\n--- 相关块 {i+1} ---")
            clean_content = doc.page_content.replace('\n', ' ').strip()
            print(clean_content[:500] + "..." if len(clean_content) > 500 else clean_content)
            print("-" * 20)
    else:
        print("未检索到相关文本块。")
    
    print(f"\n--- 模型最终回答 ---\n{result['response']}\n---")