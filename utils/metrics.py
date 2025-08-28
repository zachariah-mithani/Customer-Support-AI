import time
import json
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class MetricsCalculator:
    def __init__(self):
        self.metrics_data = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'response_times': [],
            'classification_accuracy': 0.0,
            'retrieval_accuracy': 0.0,
            'overall_accuracy': 0.0,
            'average_response_time': 0.0,
            'test_results': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_classification_accuracy(self, test_results: List[Dict]) -> float:
        """Calculate classification accuracy"""
        if not test_results:
            return 0.0
        
        correct = 0
        total = len(test_results)
        
        for result in test_results:
            if result.get('classification_correct', False):
                correct += 1
        
        return (correct / total) * 100
    
    def calculate_retrieval_accuracy(self, test_results: List[Dict]) -> float:
        """Calculate retrieval accuracy based on content relevance"""
        if not test_results:
            return 0.0
        
        relevant_count = 0
        total = len(test_results)
        
        for result in test_results:
            if result.get('retrieval_relevant', False):
                relevant_count += 1
        
        return (relevant_count / total) * 100
    
    def calculate_response_time_stats(self, response_times: List[float]) -> Dict[str, float]:
        """Calculate response time statistics"""
        if not response_times:
            return {
                'average': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            }
        
        return {
            'average': np.mean(response_times),
            'min': np.min(response_times),
            'max': np.max(response_times),
            'median': np.median(response_times),
            'std_dev': np.std(response_times)
        }
    
    def evaluate_single_query(self, query: str, expected_category: str, 
                            expected_answer_contains: List[str], 
                            actual_category: str, actual_answer: str, 
                            response_time: float) -> Dict[str, Any]:
        """Evaluate a single query result"""
        # Check classification accuracy
        classification_correct = actual_category.lower() == expected_category.lower()
        
        # Check retrieval relevance (if answer contains expected keywords)
        retrieval_relevant = False
        if actual_answer and expected_answer_contains:
            answer_lower = actual_answer.lower()
            matches = sum(1 for keyword in expected_answer_contains 
                         if keyword.lower() in answer_lower)
            retrieval_relevant = matches >= len(expected_answer_contains) * 0.5  # 50% keyword match
        
        # Overall success
        overall_success = classification_correct and retrieval_relevant
        
        return {
            'query': query,
            'expected_category': expected_category,
            'actual_category': actual_category,
            'expected_keywords': expected_answer_contains,
            'actual_answer': actual_answer,
            'response_time': response_time,
            'classification_correct': classification_correct,
            'retrieval_relevant': retrieval_relevant,
            'overall_success': overall_success
        }
    
    def process_test_results(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Process and calculate comprehensive metrics"""
        self.metrics_data['total_queries'] = len(test_results)
        self.metrics_data['test_results'] = test_results
        
        # Calculate success/failure counts
        successful = sum(1 for r in test_results if r.get('overall_success', False))
        self.metrics_data['successful_queries'] = successful
        self.metrics_data['failed_queries'] = len(test_results) - successful
        
        # Calculate accuracies
        self.metrics_data['classification_accuracy'] = self.calculate_classification_accuracy(test_results)
        self.metrics_data['retrieval_accuracy'] = self.calculate_retrieval_accuracy(test_results)
        self.metrics_data['overall_accuracy'] = (successful / len(test_results)) * 100 if test_results else 0
        
        # Calculate response time statistics
        response_times = [r.get('response_time', 0) for r in test_results]
        self.metrics_data['response_times'] = response_times
        time_stats = self.calculate_response_time_stats(response_times)
        self.metrics_data['average_response_time'] = time_stats['average']
        self.metrics_data['response_time_stats'] = time_stats
        
        return self.metrics_data
    
    def generate_report(self) -> str:
        """Generate a formatted metrics report"""
        report = f"""
# Customer Support System Metrics Report
Generated on: {self.metrics_data['timestamp']}

## Overall Performance
- **Total Queries Processed**: {self.metrics_data['total_queries']}
- **Successful Queries**: {self.metrics_data['successful_queries']}
- **Failed Queries**: {self.metrics_data['failed_queries']}
- **Overall Accuracy**: {self.metrics_data['overall_accuracy']:.1f}%

## Detailed Metrics
- **Classification Accuracy**: {self.metrics_data['classification_accuracy']:.1f}%
- **Retrieval Accuracy**: {self.metrics_data['retrieval_accuracy']:.1f}%
- **Average Response Time**: {self.metrics_data['average_response_time']:.2f} seconds

## Response Time Statistics
- **Average**: {self.metrics_data.get('response_time_stats', {}).get('average', 0):.2f}s
- **Minimum**: {self.metrics_data.get('response_time_stats', {}).get('min', 0):.2f}s
- **Maximum**: {self.metrics_data.get('response_time_stats', {}).get('max', 0):.2f}s
- **Median**: {self.metrics_data.get('response_time_stats', {}).get('median', 0):.2f}s
- **Standard Deviation**: {self.metrics_data.get('response_time_stats', {}).get('std_dev', 0):.2f}s

## Performance Summary
{'✅ Excellent performance' if self.metrics_data['overall_accuracy'] >= 90 else '⚠️ Good performance' if self.metrics_data['overall_accuracy'] >= 80 else '❌ Needs improvement'}
"""
        return report
    
    def save_metrics(self, filename: str = "metrics_report.json") -> None:
        """Save metrics to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.metrics_data, f, indent=2)
            print(f"Metrics saved to {filename}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
