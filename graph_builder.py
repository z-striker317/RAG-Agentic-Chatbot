# Graph builder for LangGraph workflow

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.reactnode import RAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""

    def __init__(self, retriever, llm):
        """Initialize the graph builder with retriever and LLM"""
        
        self.nodes = RAGNodes(retriever=retriever, llm=llm) 
        self.graph = None

    def build(self):
        """Build the graph with defined nodes and edges"""

        #Create state graph
        builder = StateGraph(RAGState) 

        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        # Set Entry point
        builder.set_entry_point("retriever")

        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        # Compile the graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question:str)-> dict:
        """ Run the RAG workflow"""

        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)