from typing import Dict, Any
import time
from config import CLASSIFICATION_CATEGORIES

class ClassificationAgent:
    def __init__(self):
        self.name = "Classification Agent"
        self.description = "Classifies customer queries into predefined categories"
        self.categories = CLASSIFICATION_CATEGORIES
        self.classifier = True  # Set to True to indicate initialized
        self.initialize_model()
    
    def initialize_model(self) -> None:
        """Initialize the classification model"""
        try:
            # Use keyword-based classification (more reliable for this demo)
            self.classifier = True
            print(f"✅ {self.name} initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing {self.name}: {e}")
            self.classifier = None
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify a customer query into predefined categories"""
        start_time = time.time()
        
        if not self.classifier:
            return {
                'category': 'general_inquiry',
                'confidence': 0.0,
                'processing_time': 0.0,
                'error': 'Classifier not initialized'
            }
        
        try:
            # Simple keyword-based classification as fallback
            # This is more reliable than using the generative model for classification
            query_lower = query.lower()
            
            # Define keywords for each category
            category_keywords = {
                'refund': ['return', 'refund', 'money back', 'exchange', 'cancel order'],
                'technical_issue': ['login', 'password', 'website', 'error', 'crash', 'bug', 'not working'],
                'billing': ['charge', 'payment', 'credit card', 'bill', 'invoice', 'charged'],
                'shipping': ['delivery', 'shipping', 'tracking', 'package', 'arrived', 'when will'],
                'product_info': ['size', 'color', 'material', 'specification', 'details', 'available'],
                'account_management': ['account', 'profile', 'email', 'address', 'settings', 'change'],
                'general_inquiry': ['help', 'support', 'question', 'info', 'hours', 'contact']
            }
            
            # Calculate scores for each category
            scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in query_lower)
                scores[category] = score
            
            # Get the category with highest score
            if max(scores.values()) > 0:
                best_category = max(scores, key=scores.get)
                confidence = scores[best_category] / len(category_keywords[best_category])
            else:
                best_category = 'general_inquiry'
                confidence = 0.5
            
            processing_time = time.time() - start_time
            
            return {
                'category': best_category,
                'confidence': min(confidence, 1.0),
                'processing_time': processing_time,
                'scores': scores,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'category': 'general_inquiry',
                'confidence': 0.0,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent"""
        return {
            'name': self.name,
            'description': self.description,
            'categories': self.categories,
            'status': 'active' if self.classifier else 'inactive',
            'model_ready': self.classifier is not None
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main processing method for the agent"""
        print(f"🤖 {self.name} processing query: '{query[:50]}...'")
        
        result = self.classify_query(query)
        
        print(f"📊 Classification result: {result['category']} (confidence: {result['confidence']:.2f})")
        
        return {
            'agent': self.name,
            'input': query,
            'output': result,
            'timestamp': time.time()
        }
