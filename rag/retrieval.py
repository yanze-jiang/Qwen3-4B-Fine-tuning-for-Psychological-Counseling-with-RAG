import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class PsyRetriever:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 确保路径指向 index.faiss 所在的文件夹
        db_path = os.path.join(current_dir, "vector_store/psychology_db")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        
        try:
            # 增加 check_path 逻辑
            if not os.path.exists(os.path.join(db_path, "index.faiss")):
                print(f"❌ 严重错误：在 {db_path} 没找到 index.faiss 文件！")
                self.vector_store = None
                return

            self.vector_store = FAISS.load_local(
                db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("✅ 心理学专业知识库已加载")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            self.vector_store = None

    def get_relevant_context(self, query):
        if not self.vector_store: return "库未加载"

        # 【调试建议】先暂时注释掉门控逻辑，强制检索看有没有结果
        # print(f"正在检索: {query}")
        
        # 使用 similarity_search_with_score 可以看到匹配分值
        # 分值越小表示越相似
        docs_with_score = self.vector_store.similarity_search_with_score(query, k=1)
        
        if not docs_with_score:
            return "检索结果为空"
            
        doc, score = docs_with_score[0]
        # print(f"匹配分值 (Score): {score}") # 调试用
        return doc.page_content

if __name__ == "__main__":
    retriever = PsyRetriever()
    # 换一个 PDF 里肯定有的词试试
    test_query = "认知疗法" 
    context = retriever.get_relevant_context(test_query)
    print(f"\n测试查询: {test_query}")
    print(f"检索内容预览: {context[:200]}")