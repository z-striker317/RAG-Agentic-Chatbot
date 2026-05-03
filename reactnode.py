# LangGraph nodes for RAG workflow + React Agent inside generate_content

import uuid
_ = uuid
from typing import List, Optional
from src.state.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class RAGNodes:
    """Contains the node functions for RAG workflow"""

    def __init__(self, retriever,llm):
        self.retriever = retriever
        self.llm = llm
        self.agent = None

    def retrieve_docs(self, state:RAGState)-> RAGState:
        """Classic Retriever Node"""
        docs=self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    # Build Tools
    # Build Tools
    def _build_tools(self):
        """Build retriever + wikipedia tools"""
        
        def retriever_tool_fn(query:str)-> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            merged=[]
            for i,d in enumerate(docs[:8],start=1):
                meta=d.metadata if hasattr(d,"metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)
        
        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed vectorstore",
            func=retriever_tool_fn
        )

        # The Wikipedia object is ALREADY a fully formatted tool
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        # We pass 'wiki' directly instead of wrapping it in another Tool()
        return [retriever_tool, wiki]


    # Build Agent
    def _build_agent(self):
        """React Agent with Tools"""
        tools= self._build_tools()
        system_prompt =(
            "You are a helpful RAG Agent"
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final answer. "
        )
        self.agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state:RAGState)->RAGState:
        """Generate answer using React Agent with retriever + wikipedia. """
        if self.agent is None:
            self._build_agent()
        
        result = self.agent.invoke({"messages": [HumanMessage(content=state.question)]})
        
        messages= result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate an answer."
        )