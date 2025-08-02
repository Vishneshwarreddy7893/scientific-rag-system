# Scientific Literature RAG System

A domain-specific Retrieval-Augmented Generation (RAG) system designed for scientific literature analysis with advanced support for technical terminology, mathematical equations, and research-backed responses with proper citations.

## 🎯 Project Overview

This RAG system specializes in processing scientific documents, particularly in the biology domain, providing intelligent question-answering capabilities with contextual understanding of research papers, technical concepts, and scientific methodologies.

## 🚀 Key Features

### Core Capabilities
- **Domain-Specific Processing**: Optimized for scientific literature with specialized terminology handling
- **Mathematical Equation Recognition**: Advanced parsing and interpretation of scientific notation and formulas
- **Citation Management**: Automatic research paper citation formatting and reference tracking
- **Evidence-Based Responses**: Research-backed answers with proper source attribution
- **Technical Accuracy Validation**: Built-in mechanisms to ensure scientific precision

### Advanced Features
- **Intelligent Document Processing**: Multi-format support (PDF, text) with scientific content extraction
- **Vector-Based Retrieval**: ChromaDB integration for efficient similarity search
- **Contextual Generation**: OpenAI GPT-3.5-turbo integration with fallback template system
- **Scientific Entity Extraction**: Automatic identification of equations, citations, and technical terms
- **Confidence Scoring**: Response reliability assessment based on retrieval quality
- **Interactive Web Interface**: Streamlit-based dashboard with real-time processing

## 🏗️ System Architecture

```
scientific-rag/
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # PDF processing and text extraction
│   ├── embeddings.py           # Vector embeddings and similarity search
│   ├── generator.py            # Response generation with OpenAI integration
│   ├── retriever.py            # Document retrieval and ranking
│   └── utils.py                # Utility functions and helpers
├── chroma_db/                  # Vector database storage
├── data/                       # Document storage directory
├── venv/                       # Virtual environment
├── app.py                      # Main Streamlit application
├── requirements.txt            # Project dependencies
├── test_setup.py              # System testing and validation
├── .env                       # Environment variables (API keys)
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (optional, system works without it)
- Minimum 8GB RAM recommended for optimal performance

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd scientific-rag
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Required Models
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
```

### Step 5: Environment Configuration
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

### Step 6: Initialize System
```bash
python test_setup.py
```

## 🎮 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the System

1. **Document Upload**: 
   - Upload PDF scientific papers through the sidebar
   - System automatically processes and indexes documents
   - Supports multiple document formats

2. **Query Processing**:
   - Enter scientific questions in natural language
   - System retrieves relevant context from uploaded documents
   - Generates evidence-based responses with citations

3. **Advanced Features**:
   - Configure OpenAI API key for enhanced responses
   - Adjust retrieval parameters in sidebar
   - View processing statistics and system status

## 🔧 Technical Implementation

### Document Processing Pipeline
The system employs a sophisticated multi-stage processing approach:

1. **Text Extraction**: Advanced PDF parsing with scientific content preservation
2. **Content Chunking**: Strategic segmentation maintaining context integrity
3. **Scientific Entity Recognition**: Automatic identification of formulas, citations, and technical terms
4. **Vector Embedding**: Sentence-transformer based semantic encoding
5. **Database Indexing**: ChromaDB storage with optimized retrieval

### Retrieval Mechanism
- **Semantic Search**: Vector similarity using sentence-transformers
- **Context Ranking**: Multi-factor scoring including relevance and scientific accuracy
- **Source Attribution**: Automatic citation generation and reference tracking

### Generation Strategy
- **Primary Mode**: OpenAI GPT-3.5-turbo with scientific domain prompting
- **Fallback Mode**: Template-based generation for offline operation
- **Quality Assurance**: Response validation and confidence scoring

## 📊 Performance Metrics

### Evaluation Framework
The system implements comprehensive evaluation using multiple metrics:

- **Retrieval Accuracy**: Semantic similarity and relevance scoring
- **Response Quality**: Technical accuracy and coherence assessment
- **Latency Measurement**: Processing time optimization
- **Citation Precision**: Reference accuracy and completeness

### Benchmarking Results
- Average retrieval latency: <2 seconds
- Document processing: ~1MB/minute
- Query response time: 3-8 seconds (depending on complexity)

## 🔬 Scientific Domain Optimization

### Biology Specialization
The system is specifically optimized for biological literature:

- **Terminology Recognition**: Comprehensive biological entity extraction
- **Methodology Understanding**: Research design and experimental procedure analysis
- **Citation Standards**: Academic formatting compliance (APA, Nature, etc.)
- **Cross-Reference Validation**: Inter-document relationship analysis

### Technical Capabilities
- Mathematical equation parsing and LaTeX interpretation
- Scientific notation standardization
- Taxonomic classification understanding
- Experimental data interpretation

## 🚦 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB storage space
- Internet connection (for model downloads)

### Recommended Specifications
- Python 3.9+
- 8GB RAM
- 5GB storage space
- OpenAI API access for optimal performance

## 🔧 Configuration Options

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key           # OpenAI integration
CHROMA_PERSIST_DIRECTORY=./chroma_db  # Database location
MAX_CHUNK_SIZE=1000                   # Document chunking size
RETRIEVAL_TOP_K=5                     # Number of retrieved contexts
```

### Model Configuration
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Generation Model**: gpt-3.5-turbo (with fallback)
- **Vector Database**: ChromaDB with persistent storage
- **Text Processing**: spaCy with scientific extensions

## 🧪 Testing & Validation

### Automated Testing
```bash
python test_setup.py
```

This validates:
- Dependency installation
- Model availability
- Database connectivity
- API configuration
- System performance

### Manual Testing
1. Upload sample scientific papers
2. Test various query types (definitions, explanations, comparisons)
3. Verify citation accuracy and formatting
4. Check response quality and relevance

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Follow PEP 8 coding standards
4. Add comprehensive tests for new features
5. Update documentation as needed
6. Submit pull request with detailed description

### Code Structure Guidelines
- Modular design with clear separation of concerns
- Comprehensive docstrings and type hints
- Error handling and logging throughout
- Configuration management via environment variables

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support & Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [your-email@domain.com]
- Documentation: [project-docs-url]

## 🔄 Version History

### v1.0.0 (Current)
- Initial release with core RAG functionality
- Scientific document processing pipeline
- OpenAI integration with fallback system
- Web-based interface with Streamlit
- Comprehensive evaluation framework

### Planned Features
- Multi-domain support (chemistry, physics)
- Advanced visualization capabilities
- API endpoint development
- Mobile-responsive interface
- Collaborative annotation system

---

**Note**: This system is designed for research and educational purposes. Always verify scientific claims with primary sources and domain experts.