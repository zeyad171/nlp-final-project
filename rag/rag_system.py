"""
==============================================
RAG SYSTEM WRAPPER
==============================================
Simple interface to the modular RAG system.
Uses the improved rag/ package with:
- Smart chunking
- Persistent FAISS index
- Gemini LLM integration
- Game-aware fallback

Author: NLP Final Project
==============================================
"""

import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Wrapper for the modular RAG system.
    
    Provides a simple interface for the Flask app:
    - generate_response(query, game_state) -> (message, intent)
    """
    
    def __init__(self):
        """Initialize the RAG system."""
        logger.info("Initializing RAG System...")
        
        # Import and create the RAG chain
        from rag.rag_chain import get_rag_chain
        self._chain = get_rag_chain()
        
        logger.info("RAG System ready!")
    
    def generate_response(self, query: str, game_state: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate a chatbot response using RAG.
        
        Args:
            query: User's question
            game_state: Dict with currentRoom, inventory, puzzlesSolved, etc.
        
        Returns:
            Tuple of (message, intent)
        """
        return self._chain.generate_response(query, game_state)
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant documentation chunks.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            Formatted context string
        """
        chunks = self._chain.retrieve(query, top_k)
        return "\n\n".join([c['content'] for c in chunks])
    
    def rebuild_index(self):
        """Force rebuild the vector index."""
        self._chain.rebuild_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return self._chain.get_stats()


# ==============================================
# SINGLETON INSTANCE
# ==============================================

_rag_instance = None

def get_rag_system() -> RAGSystem:
    """Get or create the RAG system singleton."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    return _rag_instance
