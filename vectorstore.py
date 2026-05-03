from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStore:
    """Manages vector store application"""
    def __init__(self):
        self.embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore=None
        self.retriever=None

    def create_vectorstore(self, documents:List[Document]):
        """ It will create a vector store from documents and a retriever for querying the vector store"""
        self.vectorstore=FAISS.from_documents(documents, self.embedding)
        self.retriever=self.vectorstore.as_retriever()

    def get_retriever(self):
        """Get retriever instance"""
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Please call create_vectorstore first.")
        return self.retriever
    
    def retrieve(self, query:str, k:int=4)->List[Document]:
        """ It will retrieve relevant documents for a query"""
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Please call create_vectorstore first.")
        return self.retriever.invoke(query)