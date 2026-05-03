# RAGState definition for RAG Workflow

from typing import List
from langchain_core.documents import Document
from pydantic import BaseModel

class RAGState(BaseModel):
    """State object for RAG Workflow"""

    question: str
    retrieved_docs: List[Document]=[]
    answer: str=""