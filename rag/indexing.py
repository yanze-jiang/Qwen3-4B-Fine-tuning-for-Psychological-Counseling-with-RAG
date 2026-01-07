import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_psychology_index():
    # 获取当前脚本所在的目录 (rag/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 自动定位到正确的路径
    kb_path = os.path.join(current_dir, "knowledge_base")
    save_path = os.path.join(current_dir, "vector_store/psychology_db")
    
    if not os.path.exists(kb_path):
        print(f"❌ 错误：找不到目录 {kb_path}")
        return

    # 1. 自动扫描所有 PDF
    pdf_files = [f for f in os.listdir(kb_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️ 警告：在 {kb_path} 中没有找到任何 PDF 文件")
        return
        
    all_docs = []
    
    # 针对 4B 模型的切分策略：短而精
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,        
        chunk_overlap=40,      
        length_function=len
    )

    print(f"🚀 开始处理 {len(pdf_files)} 本专业书籍...")

    for pdf in pdf_files:
        pdf_path = os.path.join(kb_path, pdf)
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split(text_splitter)
            all_docs.extend(pages)
            print(f"✅ 已完成: {pdf} (切分数量: {len(pages)})")
        except Exception as e:
            print(f"❌ 处理文件 {pdf} 时出错: {e}")

    # 2. Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'} 
    )

    # 3. 构建向量库
    print("🧠 正在构建向量索引（这可能需要几分钟，取决于 PDF 大小）...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    
    # 4. 保存
    vectorstore.save_local(save_path)
    print(f"✨ 成功！索引已保存至 {save_path}")

if __name__ == "__main__":
    build_psychology_index()