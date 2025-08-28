from typing import Dict, List, Any
import time

class ResponseAgent:
    def __init__(self):
        self.name = "Response Agent"
        self.description = "Generates and validates customer support responses"
        self.generator = True  # Set to True to indicate initialized
        self.initialize_model()
    
    def initialize_model(self) -> None:
        """Initialize the response generation model"""
        try:
            # Use rule-based response generation for this demo
            self.generator = True
            print(f"✅ {self.name} initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing {self.name}: {e}")
            self.generator = None
    
    def generate_response(self, query: str, retrieved_faqs: List[Dict], category: str) -> Dict[str, Any]:
        """Generate a response based on query and retrieved FAQ data"""
        start_time = time.time()
        
        try:
            # Use the most relevant FAQ as the base response
            if retrieved_faqs:
                base_response = retrieved_faqs[0]['answer']
                confidence = retrieved_faqs[0]['similarity_score']
                
                # Enhance the response with context
                enhanced_response = self.enhance_response(query, base_response, category)
                
                processing_time = time.time() - start_time
                
                return {
                    'response': enhanced_response,
                    'base_faq': retrieved_faqs[0],
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'sources': len(retrieved_faqs),
                    'error': None
                }
            else:
                # Generate a fallback response
                fallback_response = self.generate_fallback_response(query, category)
                
                processing_time = time.time() - start_time
                
                return {
                    'response': fallback_response,
                    'base_faq': None,
                    'confidence': 0.3,
                    'processing_time': processing_time,
                    'sources': 0,
                    'error': None
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please contact our customer support team directly for assistance.",
                'base_faq': None,
                'confidence': 0.0,
                'processing_time': processing_time,
                'sources': 0,
                'error': str(e)
            }
    
    def enhance_response(self, query: str, base_response: str, category: str) -> str:
        """Enhance the base response with personalization and context"""
        # Add greeting and personalization
        greeting = "Thank you for contacting our support team. "
        
        # Add category-specific context
        category_context = {
            'refund': "I understand you have a question about returns or refunds. ",
            'technical_issue': "I see you're experiencing a technical issue. ",
            'billing': "I can help you with your billing inquiry. ",
            'shipping': "I'll help you with your shipping question. ",
            'product_info': "I'm happy to provide product information. ",
            'account_management': "I can assist you with your account. ",
            'general_inquiry': "I'm here to help with your inquiry. "
        }
        
        context = category_context.get(category, "")
        
        # Combine and format the response
        enhanced_response = f"{greeting}{context}{base_response}"
        
        # Add closing
        closing = " If you have any additional questions or need further assistance, please don't hesitate to reach out to our customer support team."
        
        return enhanced_response + closing
    
    def generate_fallback_response(self, query: str, category: str) -> str:
        """Generate a fallback response when no relevant FAQ is found"""
        fallback_responses = {
            'refund': "I understand you have a question about returns or refunds. While I don't have specific information about your inquiry, I recommend contacting our customer service team at your earliest convenience. They can provide detailed assistance with your refund or return request.",
            'technical_issue': "I see you're experiencing a technical issue. For technical problems, I recommend trying to clear your browser cache and cookies, or try using a different browser. If the issue persists, please contact our technical support team with details about your device and browser.",
            'billing': "I can help you with billing inquiries. For specific billing questions, please contact our billing department with your order number and account information. They can investigate and resolve any billing concerns you may have.",
            'shipping': "I'll help you with shipping questions. For specific shipping inquiries, please check your email for tracking information or log into your account to view order status. If you need further assistance, our support team can help track your order.",
            'product_info': "I'm happy to help with product information. For specific product details, please visit our website or contact our product specialists who can provide detailed information about availability, specifications, and features.",
            'account_management': "I can assist with account-related questions. For account changes or issues, please log into your account settings or contact our customer service team. They can help with password resets, address changes, and other account modifications.",
            'general_inquiry': "Thank you for your inquiry. While I don't have specific information about your question, our customer service team is available to provide personalized assistance. Please contact them with your specific needs."
        }
        
        base_response = fallback_responses.get(category, fallback_responses['general_inquiry'])
        
        return f"Thank you for contacting our support team. {base_response} Our customer service hours are Monday through Friday, 9 AM to 6 PM EST."
    
    def validate_response(self, response: str, query: str) -> Dict[str, Any]:
        """Validate the generated response"""
        validation_result = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # Check response length
        if len(response) < 50:
            validation_result['issues'].append("Response too short")
            validation_result['quality_score'] -= 0.2
        elif len(response) > 1000:
            validation_result['issues'].append("Response too long")
            validation_result['quality_score'] -= 0.1
        
        # Check for greeting
        if not any(greeting in response.lower() for greeting in ['thank you', 'hello', 'hi']):
            validation_result['suggestions'].append("Consider adding a greeting")
            validation_result['quality_score'] -= 0.1
        
        # Check for closing
        if not any(closing in response.lower() for closing in ['contact', 'help', 'assistance', 'support']):
            validation_result['suggestions'].append("Consider adding contact information")
            validation_result['quality_score'] -= 0.1
        
        # Base quality score
        validation_result['quality_score'] = max(0.0, min(1.0, 0.8 + validation_result['quality_score']))
        
        # Overall validation
        validation_result['is_valid'] = len(validation_result['issues']) == 0
        
        return validation_result
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent"""
        return {
            'name': self.name,
            'description': self.description,
            'model_ready': self.generator is not None,
            'status': 'active' if self.generator else 'inactive'
        }
    
    def process_query(self, query: str, retrieved_faqs: List[Dict], category: str) -> Dict[str, Any]:
        """Main processing method for the agent"""
        print(f"✍️ {self.name} generating response for: '{query[:50]}...'")
        
        # Generate response
        generation_result = self.generate_response(query, retrieved_faqs, category)
        
        # Validate response
        validation_result = self.validate_response(generation_result['response'], query)
        
        print(f"📝 Generated response (quality: {validation_result['quality_score']:.2f})")
        
        return {
            'agent': self.name,
            'input': {
                'query': query,
                'category': category,
                'retrieved_faqs': len(retrieved_faqs)
            },
            'output': {
                'generation': generation_result,
                'validation': validation_result
            },
            'timestamp': time.time()
        }
