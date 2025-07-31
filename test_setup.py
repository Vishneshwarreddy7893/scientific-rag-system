#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly
"""

def test_imports():
    """Test if all required libraries can be imported"""
    
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit import failed")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError:
        print("❌ PyPDF2 import failed")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")
    except ImportError:
        print("❌ Sentence Transformers import failed")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB imported successfully")
    except ImportError:
        print("❌ ChromaDB import failed")
        return False
    
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError:
        print("❌ NLTK import failed")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError:
        print("❌ Pandas import failed")
        return False
    
    try:
        import numpy as np
        print("✅ Numpy imported successfully")
    except ImportError:
        print("❌ Numpy import failed")
        return False
    
    return True

def test_model_download():
    """Test if we can load a sentence transformer model"""
    
    print("\nTesting model download...")
    
    try:
        from sentence_transformers import SentenceTransformer
        # Use a small, fast model for testing
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformer model loaded successfully")
        
        # Test encoding
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text)
        print(f"✅ Text encoding successful. Embedding shape: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_chromadb():
    """Test ChromaDB functionality"""
    
    print("\nTesting ChromaDB...")
    
    try:
        import chromadb
        client = chromadb.Client()
        
        # Create a test collection
        collection = client.create_collection("test_collection")
        print("✅ ChromaDB collection created successfully")
        
        # Clean up
        client.delete_collection("test_collection")
        print("✅ ChromaDB cleanup successful")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running setup tests...\n")
    
    # Run all tests
    imports_ok = test_imports()
    model_ok = test_model_download()
    chromadb_ok = test_chromadb()
    
    print("\n" + "="*50)
    
    if imports_ok and model_ok and chromadb_ok:
        print("🎉 All tests passed! Your setup is ready.")
        print("You can now proceed to Phase 2: Data Collection")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("You may need to reinstall some packages or check your Python version.")
    
    print("="*50)