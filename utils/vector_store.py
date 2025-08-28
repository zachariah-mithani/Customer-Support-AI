import json
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple
from config import FAISS_INDEX_PATH, VECTOR_DIMENSION
import hashlib

class VectorStore:
    def __init__(self):
        self.index = None
        self.faq_data = []
        self.embeddings = None
        self.vocab = []
        self.word_to_idx = {}
        
    def load_faq_data(self, faq_file_path: str) -> None:
        """Load FAQ data from JSON file"""
        try:
            with open(faq_file_path, 'r') as f:
                self.faq_data = json.load(f)
            print(f"Loaded {len(self.faq_data)} FAQ entries")
        except Exception as e:
            print(f"Error loading FAQ data: {e}")
            self.faq_data = []
    
    def create_embeddings(self) -> None:
        """Create embeddings for all FAQ entries"""
        if not self.faq_data:
            print("No FAQ data loaded")
            return
            
        # Create simple TF-IDF style embeddings (word frequency based)
        texts = []
        for faq in self.faq_data:
            combined_text = f"{faq['question']} {faq['answer']}"
            texts.append(combined_text.lower())
        
        print("Creating embeddings...")
        self.embeddings = self._create_simple_embeddings(texts)
        print(f"Created {len(self.embeddings)} embeddings")
    
    def _create_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create simple word-based embeddings"""
        # Create vocabulary if not exists
        if not self.vocab:
            vocab = set()
            for text in texts:
                words = text.split()
                vocab.update(words)
            
            self.vocab = sorted(list(vocab))
            self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        
        # Create embeddings matrix
        embeddings = np.zeros((len(texts), min(len(self.vocab), VECTOR_DIMENSION)))
        
        for i, text in enumerate(texts):
            words = text.split()
            for word in words:
                if word in self.word_to_idx:
                    idx = self.word_to_idx[word]
                    if idx < VECTOR_DIMENSION:
                        embeddings[i, idx] += 1
        
        # Normalize embeddings
        for i in range(len(embeddings)):
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] = embeddings[i] / norm
        
        return embeddings
    
    def build_index(self) -> None:
        """Build FAISS index from embeddings"""
        if self.embeddings is None:
            print("No embeddings available. Create embeddings first.")
            return
            
        try:
            # Create FAISS index - use L2 distance for better compatibility
            embedding_dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(embedding_dim)
            
            # Add embeddings to index (they're already normalized)
            self.index.add(self.embeddings.astype('float32'))
            
            print(f"Built FAISS index with {self.index.ntotal} entries")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.index = None
    
    def save_index(self) -> None:
        """Save FAISS index and associated data"""
        if self.index is None:
            print("No index to save")
            return
            
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{FAISS_INDEX_PATH}.index")
            
            # Save FAQ data and embeddings
            with open(f"{FAISS_INDEX_PATH}_data.pkl", 'wb') as f:
                pickle.dump({
                    'faq_data': self.faq_data,
                    'embeddings': self.embeddings,
                    'vocab': self.vocab,
                    'word_to_idx': self.word_to_idx
                }, f)
            
            print("Index and data saved successfully")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self) -> bool:
        """Load FAISS index and associated data"""
        try:
            # Load FAISS index
            if os.path.exists(f"{FAISS_INDEX_PATH}.index"):
                self.index = faiss.read_index(f"{FAISS_INDEX_PATH}.index")
                
                # Load FAQ data and embeddings
                with open(f"{FAISS_INDEX_PATH}_data.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.faq_data = data['faq_data']
                    self.embeddings = data['embeddings']
                    self.vocab = data.get('vocab', [])
                    self.word_to_idx = data.get('word_to_idx', {})
                
                print(f"Index loaded with {self.index.ntotal} entries")
                return True
            else:
                print("No existing index found")
                return False
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar FAQ entries"""
        if self.index is None:
            print("No index available. Build index first.")
            return []
        
        try:
            # Create query embedding using simple approach
            query_embedding = self._create_simple_embeddings([query.lower()])
            
            # Search in index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.faq_data):
                    result = {
                        'rank': i + 1,
                        'score': float(score),
                        'faq': self.faq_data[idx]
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def initialize(self, faq_file_path: str = "data/faq_dataset.json") -> None:
        """Initialize the vector store with FAQ data"""
        # Try to load existing index
        if not self.load_index():
            # If no index exists, create new one
            self.load_faq_data(faq_file_path)
            self.create_embeddings()
            self.build_index()
            self.save_index()
        
        print("Vector store initialized successfully")
