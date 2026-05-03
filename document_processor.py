# Document processing module for loading and splitting documents into manageable chunks for embedding and indexing.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

class DocumentProcessor:
    # Handle Document loading and processing

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """"Initialize the DocumentProcessor with specified chunk size and overlap.

        Args:
            chunk_size (int): The size of each text chunk for embedding.
            chunk_overlap (int): The number of overlapping characters between chunks.
        """

        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_from_url(self, url:str)->List[Document]:
        """ It will load documents from urls"""
        loader=WebBaseLoader(url)
        return loader.load()
    
    def load_from_pdf_dir(self, directory: Union[str,Path])->List[Document]: #Union allows a string that is the path to directory
        """ It will load documents from a directory containing PDF files"""
        loader=PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_from_txt(self, file_path: Union[str,Path])->List[Document]:
        """ It will load documents from a text file"""
        loader=TextLoader(str(file_path), encoding="utf-8")
        return loader.load()
    
    def load_from_pdf(self, file_path: Union[str, Path])->List[Document]:
        """ It will load documents from a PDF file"""
        loader=PyPDFLoader(str("data"))
        return loader.load()
    
    def load_documents(self, sources:List[str])->List[Document]:
        """ It will load documents from a list of sources (URLs, PDF files, text files)"""
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))

            path=Path("Data")
            if path.is_dir(): #PDF Directory
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported file type: {src}. "
                    "Use URL, .txt file, or PDF directory."
                )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split loaded documents into chunks for embedding."""
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls:List[str])->List[Document]:
        """Load and split documents."""
        docs = self.load_documents(urls)
        return self.split_documents(docs)