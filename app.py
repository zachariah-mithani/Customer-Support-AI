import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Import agents
from agents.classification_agent import ClassificationAgent
from agents.retrieval_agent import RetrievalAgent
from agents.response_agent import ResponseAgent

# Import utilities
from utils.metrics import MetricsCalculator

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Customer Support System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.classification_agent = None
    st.session_state.retrieval_agent = None
    st.session_state.response_agent = None
    st.session_state.query_history = []
    st.session_state.metrics_calculator = MetricsCalculator()

def initialize_agents():
    """Initialize all agents"""
    if not st.session_state.agents_initialized:
        with st.spinner("Initializing AI agents..."):
            st.session_state.classification_agent = ClassificationAgent()
            st.session_state.retrieval_agent = RetrievalAgent()
            st.session_state.response_agent = ResponseAgent()
            st.session_state.agents_initialized = True
        st.success("✅ All agents initialized successfully!")

def process_customer_query(query: str) -> Dict[str, Any]:
    """Process a customer query through all agents"""
    start_time = time.time()
    
    # Step 1: Classification
    classification_result = st.session_state.classification_agent.process_query(query)
    classified_category = classification_result['output']['category']
    
    # Step 2: Retrieval
    retrieval_result = st.session_state.retrieval_agent.process_query(query, classified_category)
    retrieved_faqs = retrieval_result['output']['retrieved_faqs']
    
    # Step 3: Response Generation
    response_result = st.session_state.response_agent.process_query(query, retrieved_faqs, classified_category)
    
    total_time = time.time() - start_time
    
    return {
        'query': query,
        'total_processing_time': total_time,
        'classification': classification_result,
        'retrieval': retrieval_result,
        'response': response_result,
        'timestamp': datetime.now().isoformat()
    }

def display_agent_results(results: Dict[str, Any]):
    """Display results from all agents"""
    
    # Overview
    st.subheader("🔍 Query Processing Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Processing Time", f"{results['total_processing_time']:.2f}s")
    
    with col2:
        classification_output = results['classification']['output']
        st.metric("Classified Category", classification_output['category'])
    
    with col3:
        retrieval_output = results['retrieval']['output']
        st.metric("Retrieved FAQs", retrieval_output['relevant_results'])
    
    # Detailed results
    st.subheader("📊 Agent-by-Agent Results")
    
    # Classification Agent Results
    with st.expander("🤖 Classification Agent Results", expanded=True):
        classification_output = results['classification']['output']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Classification Details:**")
            st.write(f"- **Category:** {classification_output['category']}")
            st.write(f"- **Confidence:** {classification_output['confidence']:.2f}")
            st.write(f"- **Processing Time:** {classification_output['processing_time']:.3f}s")
        
        with col2:
            st.write("**Category Scores:**")
            if 'scores' in classification_output:
                for category, score in classification_output['scores'].items():
                    st.write(f"- {category}: {score}")
    
    # Retrieval Agent Results
    with st.expander("🔍 Retrieval Agent Results", expanded=True):
        retrieval_output = results['retrieval']['output']
        
        st.write(f"**Retrieved {retrieval_output['relevant_results']} relevant FAQ entries:**")
        
        for i, faq in enumerate(retrieval_output['retrieved_faqs'][:3]):  # Show top 3
            st.write(f"**FAQ {i+1}** (Score: {faq['similarity_score']:.3f})")
            st.write(f"- **Question:** {faq['question']}")
            st.write(f"- **Answer:** {faq['answer'][:200]}...")
            st.write(f"- **Category:** {faq['category']}")
            st.write("---")
    
    # Response Agent Results
    with st.expander("✍️ Response Agent Results", expanded=True):
        response_output = results['response']['output']
        generation = response_output['generation']
        validation = response_output['validation']
        
        st.write("**Generated Response:**")
        st.info(generation['response'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Generation Details:**")
            st.write(f"- **Confidence:** {generation['confidence']:.2f}")
            st.write(f"- **Sources Used:** {generation['sources']}")
            st.write(f"- **Processing Time:** {generation['processing_time']:.3f}s")
        
        with col2:
            st.write("**Response Validation:**")
            st.write(f"- **Quality Score:** {validation['quality_score']:.2f}")
            st.write(f"- **Valid:** {'✅' if validation['is_valid'] else '❌'}")
            if validation['issues']:
                st.write("- **Issues:** " + ", ".join(validation['issues']))
            if validation['suggestions']:
                st.write("- **Suggestions:** " + ", ".join(validation['suggestions']))

def run_test_queries():
    """Run test queries and calculate metrics"""
    
    st.subheader("🧪 Test Query Evaluation")
    
    # Load test queries
    try:
        with open('data/test_queries.json', 'r') as f:
            test_queries = json.load(f)
        st.success(f"✅ Loaded {len(test_queries)} test queries")
    except FileNotFoundError:
        st.error("Test queries file not found!")
        return
    
    # Add some space before the button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        run_tests = st.button("🚀 Run All Test Queries", key="run_test_queries_btn", type="primary")
    
    with col2:
        if st.button("🗑️ Clear Results", key="clear_test_results_btn"):
            if hasattr(st.session_state, 'test_results'):
                del st.session_state.test_results
                del st.session_state.test_metrics
                st.success("Test results cleared!")
                st.rerun()
    
    with col3:
        if st.session_state.agents_initialized:
            st.info("All agents ready for testing")
        else:
            st.error("Agents not initialized")
    
    if run_tests:
        st.markdown("---")
        st.info("🔄 Starting test query evaluation... This may take a moment.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        test_results = []
        
        for i, test_query in enumerate(test_queries):
            progress_bar.progress((i + 1) / len(test_queries))
            status_text.text(f"Processing query {i+1}/{len(test_queries)}: {test_query['query'][:50]}...")
            
            # Process query
            query_result = process_customer_query(test_query['query'])
            
            # Extract results
            actual_category = query_result['classification']['output']['category']
            actual_answer = query_result['response']['output']['generation']['response']
            response_time = query_result['total_processing_time']
            
            # Evaluate result
            evaluation = st.session_state.metrics_calculator.evaluate_single_query(
                test_query['query'],
                test_query['expected_category'],
                test_query['expected_answer_contains'],
                actual_category,
                actual_answer,
                response_time
            )
            
            test_results.append(evaluation)
        
        status_text.text("✅ All test queries completed! Processing results...")
        
        # Calculate comprehensive metrics
        metrics = st.session_state.metrics_calculator.process_test_results(test_results)
        
        # Store results in session state
        st.session_state.test_results = test_results
        st.session_state.test_metrics = metrics
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        with results_container:
            st.subheader("📈 Test Results Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.1f}%")
            
            with col2:
                st.metric("Classification Accuracy", f"{metrics['classification_accuracy']:.1f}%")
            
            with col3:
                st.metric("Retrieval Accuracy", f"{metrics['retrieval_accuracy']:.1f}%")
            
            with col4:
                st.metric("Avg Response Time", f"{metrics['average_response_time']:.2f}s")
            
            # Detailed metrics
            st.subheader("📊 Detailed Metrics")
            
            # Response time distribution
            st.write("**Response Time Statistics:**")
            time_stats = metrics['response_time_stats']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"- **Min:** {time_stats['min']:.2f}s")
                st.write(f"- **Max:** {time_stats['max']:.2f}s")
            
            with col2:
                st.write(f"- **Median:** {time_stats['median']:.2f}s")
                st.write(f"- **Std Dev:** {time_stats['std_dev']:.2f}s")
            
            with col3:
                st.write(f"- **Successful:** {metrics['successful_queries']}")
                st.write(f"- **Failed:** {metrics['failed_queries']}")
            
            # Individual test results
            st.subheader("🔍 Individual Test Results")
            
            for i, result in enumerate(test_results):
                status = "✅" if result['overall_success'] else "❌"
                
                with st.expander(f"{status} Test {i+1}: {result['query'][:50]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Expected:**")
                        st.write(f"- Category: {result['expected_category']}")
                        st.write(f"- Keywords: {', '.join(result['expected_keywords'])}")
                    
                    with col2:
                        st.write("**Actual:**")
                        st.write(f"- Category: {result['actual_category']}")
                        st.write(f"- Classification: {'✅' if result['classification_correct'] else '❌'}")
                        st.write(f"- Retrieval: {'✅' if result['retrieval_relevant'] else '❌'}")
                        st.write(f"- Response Time: {result['response_time']:.2f}s")
            
            # Generate report
            st.subheader("📋 Metrics Report")
            report = st.session_state.metrics_calculator.generate_report()
            st.markdown(report)
            
            # Save metrics
            st.session_state.metrics_calculator.save_metrics()
            st.success("✅ Metrics saved to metrics_report.json")
    
    # Display previous results if they exist
    if hasattr(st.session_state, 'test_results') and st.session_state.test_results:
        st.markdown("---")
        st.subheader("📊 Previous Test Results")
        
        metrics = st.session_state.test_metrics
        test_results = st.session_state.test_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.1f}%")
        
        with col2:
            st.metric("Classification Accuracy", f"{metrics['classification_accuracy']:.1f}%")
        
        with col3:
            st.metric("Retrieval Accuracy", f"{metrics['retrieval_accuracy']:.1f}%")
        
        with col4:
            st.metric("Avg Response Time", f"{metrics['average_response_time']:.2f}s")
        
        # Show individual results in an expander
        with st.expander("🔍 View Individual Test Results"):
            for i, result in enumerate(test_results):
                status = "✅" if result['overall_success'] else "❌"
                
                with st.container():
                    st.write(f"**{status} Test {i+1}:** {result['query']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"- **Expected Category:** {result['expected_category']}")
                        st.write(f"- **Actual Category:** {result['actual_category']}")
                        st.write(f"- **Classification:** {'✅' if result['classification_correct'] else '❌'}")
                    
                    with col2:
                        st.write(f"- **Response Time:** {result['response_time']:.2f}s")
                        st.write(f"- **Retrieval:** {'✅' if result['retrieval_relevant'] else '❌'}")
                        st.write(f"- **Overall:** {'✅' if result['overall_success'] else '❌'}")
                    
                    st.markdown("---")

def main():
    """Main application"""
    
    st.title("🤖 Multi-Agent Customer Support System")
    st.markdown("*Powered by LangChain, FAISS, and Hugging Face Transformers*")
    
    # Initialize agents
    initialize_agents()
    
    # Sidebar
    st.sidebar.title("🔧 System Controls")
    
    # Agent status
    st.sidebar.subheader("Agent Status")
    
    if st.session_state.agents_initialized:
        agents = [
            st.session_state.classification_agent,
            st.session_state.retrieval_agent,
            st.session_state.response_agent
        ]
        
        for agent in agents:
            info = agent.get_agent_info()
            status = "🟢 Active" if info.get('status') == 'active' or info.get('model_ready') else "🔴 Inactive"
            st.sidebar.write(f"**{info['name']}**: {status}")
    else:
        st.sidebar.write("Agents not initialized")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Query Interface", "🧪 Test & Metrics", "📚 Documentation"])
    
    with tab1:
        st.header("Customer Query Interface")
        
        # Query input
        query = st.text_area(
            "Enter your customer support query:",
            placeholder="e.g., I want to return my order from last week",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            process_query = st.button("Process Query", type="primary")
        
        with col2:
            if st.button("Clear History"):
                st.session_state.query_history = []
                st.success("Query history cleared!")
        
        if process_query and query:
            with st.spinner("Processing query through all agents..."):
                results = process_customer_query(query)
                st.session_state.query_history.append(results)
                display_agent_results(results)
        
        # Query history
        if st.session_state.query_history:
            st.subheader("📝 Query History")
            for i, historical_query in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query {len(st.session_state.query_history) - i}: {historical_query['query'][:50]}..."):
                    display_agent_results(historical_query)
    
    with tab2:
        run_test_queries()
    
    with tab3:
        st.header("📚 System Documentation")
        
        st.markdown("""
        ## Multi-Agent Customer Support System
        
        This system uses three specialized AI agents working in sequence to provide accurate customer support responses:
        
        ### 🤖 Agent 1: Classification Agent
        - **Purpose**: Classifies customer queries into predefined categories
        - **Technology**: Hugging Face Transformers with keyword-based classification
        - **Categories**: refund, technical_issue, billing, shipping, product_info, account_management, general_inquiry
        
        ### 🔍 Agent 2: Retrieval Agent
        - **Purpose**: Retrieves relevant FAQ entries using semantic search
        - **Technology**: FAISS vector database with Sentence Transformers
        - **Features**: Vector similarity search, category filtering, relevance scoring
        
        ### ✍️ Agent 3: Response Agent
        - **Purpose**: Generates and validates customer support responses
        - **Technology**: Hugging Face text generation with response validation
        - **Features**: Context-aware response generation, quality validation, personalization
        
        ### 📊 Key Features
        - **Real-time Processing**: Multi-agent coordination with live status updates
        - **Comprehensive Metrics**: Accuracy, response time, and quality measurements
        - **Vector Search**: FAISS-powered semantic search through 20 FAQ entries
        - **Response Validation**: Automated quality checks and suggestions
        - **Test Suite**: 20 test queries with automated evaluation
        
        ### 🛠️ Technical Stack
        - **Frontend**: Streamlit for interactive web interface
        - **AI Framework**: LangChain for agent orchestration
        - **Models**: Hugging Face Transformers (microsoft/DialoGPT-medium)
        - **Vector Database**: FAISS for efficient similarity search
        - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
        """)

if __name__ == "__main__":
    main()
