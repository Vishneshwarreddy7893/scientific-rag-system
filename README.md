# 🔬 Scientific Literature RAG System

A **Retrieval-Augmented Generation (RAG)** system specifically designed for scientific literature in the **Biology domain**. This system can understand technical terminology, mathematical equations, and provide research-backed answers with proper citations.

## 🎯 Features

- **📚 Document Processing**: Upload and process PDF research papers
- **🔍 Semantic Search**: Find relevant information using advanced embeddings
- **🤖 Intelligent Answers**: Generate context-aware scientific responses
- **📊 Equation Recognition**: Detect and handle mathematical equations
- **📖 Citation Support**: Provide proper citations for all answers
- **🎨 Modern UI**: Beautiful Streamlit interface with real-time feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for model loading)
- Internet connection (for initial model download)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd scientific-rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 How to Use

### Step 1: Upload Documents
1. Go to **"Document Management"** tab
2. Click **"Upload Documents"**
3. Select your biology research papers (PDF format)
4. Adjust chunk size and overlap if needed (recommended: 400 words, 100 overlap)
5. Click **"Process Documents"** and wait for completion

### Step 2: Ask Questions
1. Go to **"Search & Query"** tab
2. Type your scientific question in the text area
3. Choose search type (General, Abstract only, etc.)
4. Set number of sources (recommended: 3-5)
5. Click **"Search & Answer"**

### Step 3: Review Results
- Read the generated answer with confidence score
- Check the sources used for the answer
- View the retrieved context if needed
- Provide feedback to help improve the system

## 🔬 Scientific Domain: Biology

This system is optimized for biological research including:
- **Molecular Biology**: DNA, RNA, proteins, gene expression
- **Cell Biology**: Cellular processes, organelles, membranes
- **Genetics**: Inheritance, mutations, genetic disorders
- **Biochemistry**: Enzymes, metabolism, biochemical pathways
- **Evolutionary Biology**: Natural selection, adaptation, speciation
- **Microbiology**: Bacteria, viruses, microbial processes

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **Text Processing**: PyPDF2, NLTK, spaCy
- **Machine Learning**: scikit-learn, NumPy, Pandas
- **Generation**: Template-based with keyword matching

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Extraction│───▶│  Chunking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Answer Gen    │◀───│  Vector Search  │◀───│  Embeddings     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Key Features Explained

### Document Processing
- **Intelligent Chunking**: Splits documents into overlapping chunks while preserving context
- **Equation Detection**: Automatically identifies mathematical equations and formulas
- **Citation Extraction**: Finds and preserves citation information
- **Section Classification**: Identifies abstracts, methods, results, discussions

### Semantic Search
- **Advanced Embeddings**: Uses state-of-the-art sentence transformers
- **Similarity Scoring**: Cosine similarity for relevance ranking
- **Metadata Filtering**: Search by section type, equations, citations
- **Fallback Mechanisms**: Multiple search strategies for robust results

### Answer Generation
- **Context-Aware**: Uses retrieved context to generate relevant answers
- **Template-Based**: Reliable generation without token limits
- **Confidence Scoring**: Provides confidence levels for answers
- **Citation Integration**: Includes source citations in responses

## 📈 Performance Metrics

The system tracks various performance metrics:
- **Query Processing Time**: Average response time
- **Document Statistics**: Number of chunks, files, equations
- **Search Accuracy**: Relevance scores and result quality
- **System Health**: Memory usage, disk space, uptime

## 🔧 Configuration

### Chunking Parameters
- **Chunk Size**: 200-800 words (default: 400)
- **Overlap**: 50-200 words (default: 100)
- **Section Types**: Abstract, Methods, Results, Discussion

### Search Parameters
- **Max Results**: 1-10 sources (default: 3)
- **Search Types**: General, Abstract, Methods, Results, Equations
- **Similarity Threshold**: Configurable relevance scoring

## 🐛 Troubleshooting

### Common Issues

**"No relevant documents found"**
- Upload more documents related to your question
- Try rephrasing your question with different terms
- Reduce chunk size to 200-300 words for better granularity

**"System initialization failed"**
- Check if all required packages are installed
- Restart the application
- Check system logs for specific error messages

**"Error processing PDF"**
- Ensure PDF is not corrupted or password-protected
- Check file size (must be under 50MB)
- Ensure PDF contains extractable text

**Slow response times**
- Reduce number of search results
- Clear unused documents from database
- Check system resources (CPU/memory usage)

### Performance Tips
- **Smaller chunks** (200-400 words) often work better
- **Relevant documents** = better answers
- **Specific questions** get better results
- **Multiple sources** improve answer quality

## 📝 Sample Questions

Try these example questions after uploading biology papers:

- "What is Industrial Microbiology?"
- "Explain the role of mitochondria in cellular processes"
- "What are the changes in colony morphology during adaptation?"
- "How many proteins were identified in the study?"
- "What is DNA replication and how does it work?"
- "Describe protein synthesis mechanisms"

## 🔒 Privacy & Security

- **Local Processing**: All processing happens on your machine
- **No External APIs**: No data is sent to external services
- **Local Storage**: Documents stored locally in `data/papers` directory
- **Database Privacy**: Vector database stored locally in `chroma_db` directory

## 📚 Research Applications

This system is ideal for:
- **Literature Reviews**: Quickly find relevant papers and information
- **Research Planning**: Identify gaps and opportunities in existing research
- **Teaching Support**: Create educational content from research papers
- **Grant Writing**: Find supporting evidence for research proposals
- **Collaboration**: Share insights from large document collections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: For powerful text embeddings
- **ChromaDB**: For efficient vector storage
- **Streamlit**: For the beautiful web interface
- **PyPDF2**: For PDF text extraction
- **NLTK**: For natural language processing

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the system logs in the "System Status" tab
3. Create an issue on GitHub with detailed error information

---

**Built with ❤️ for the scientific community**
