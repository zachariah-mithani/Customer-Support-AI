from typing import Dict, List, Any
import time
from utils.vector_store import VectorStore
from config import SIMILARITY_THRESHOLD

class RetrievalAgent:
    def __init__(self):
        self.name = "Retrieval Agent"
        self.description = "Retrieves relevant FAQ entries using FAISS vector search"
        self.vector_store = VectorStore()
        self.initialize_vector_store()
    
    def initialize_vector_store(self) -> None:
        """Initialize the vector store with FAQ data"""
        try:
            self.vector_store.initialize()
            print(f"✅ {self.name} initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing {self.name}: {e}")
            import traceback
            traceback.print_exc()
    
    def retrieve_relevant_faqs(self, query: str, category: str = None, k: int = 3) -> Dict[str, Any]:
        """Retrieve relevant FAQ entries for a given query"""
        start_time = time.time()
        
        try:
            # Search for similar FAQ entries
            search_results = self.vector_store.search(query, k=k)
            
            # Filter by category if provided
            if category and category != 'general_inquiry':
                filtered_results = []
                for result in search_results:
                    if result['faq'].get('category') == category:
                        filtered_results.append(result)
                
                # If no category-specific results, use all results
                if filtered_results:
                    search_results = filtered_results
            
            # Process results
            relevant_faqs = []
            for result in search_results:
                if result['score'] >= SIMILARITY_THRESHOLD:
                    relevant_faqs.append({
                        'question': result['faq']['question'],
                        'answer': result['faq']['answer'],
                        'category': result['faq']['category'],
                        'similarity_score': result['score'],
                        'rank': result['rank']
                    })
            
            processing_time = time.time() - start_time
            
            return {
                'query': query,
                'category_filter': category,
                'retrieved_faqs': relevant_faqs,
                'total_results': len(search_results),
                'relevant_results': len(relevant_faqs),
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'query': query,
                'category_filter': category,
                'retrieved_faqs': [],
                'total_results': 0,
                'relevant_results': 0,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def get_best_match(self, query: str, category: str = None) -> Dict[str, Any]:
        """Get the single best matching FAQ entry"""
        retrieval_result = self.retrieve_relevant_faqs(query, category, k=1)
        
        if retrieval_result['relevant_results'] > 0:
            best_match = retrieval_result['retrieved_faqs'][0]
            return {
                'found': True,
                'faq': best_match,
                'confidence': best_match['similarity_score'],
                'processing_time': retrieval_result['processing_time']
            }
        else:
            return {
                'found': False,
                'faq': None,
                'confidence': 0.0,
                'processing_time': retrieval_result['processing_time']
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent"""
        return {
            'name': self.name,
            'description': self.description,
            'vector_store_ready': self.vector_store.index is not None,
            'faq_count': len(self.vector_store.faq_data),
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'status': 'active' if self.vector_store.index is not None else 'inactive'
        }
    
    def process_query(self, query: str, category: str = None) -> Dict[str, Any]:
        """Main processing method for the agent"""
        print(f"🔍 {self.name} searching for: '{query[:50]}...' (category: {category})")
        
        result = self.retrieve_relevant_faqs(query, category)
        
        print(f"📚 Retrieved {result['relevant_results']} relevant FAQ entries")
        
        return {
            'agent': self.name,
            'input': {'query': query, 'category': category},
            'output': result,
            'timestamp': time.time()
        }
