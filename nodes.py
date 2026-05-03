"""Defining the nodes for the graph. LangGraph nodes for RAG workflow"""

from src.state.rag_state import RAGState

class RAGNodes:
    """Contains the node functions for RAG workflow"""

    def __init__(self, retriever,llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state:RAGState):
        """Retrieve relevant document nodes"""
        docs=self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def generate_answer(self, state:RAGState)->RAGState:
        """Generate answer from retrieved document nodes"""
        
        # Combine retrieved docs into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Create prompt
        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {state.question}"""
        
        # Generate response
        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )