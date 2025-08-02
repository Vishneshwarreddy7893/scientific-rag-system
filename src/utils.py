"""
Utility classes for the Scientific RAG System
"""

import os
import shutil
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import platform
import psutil

class FileManager:
    def __init__(self, data_dir: str = "data/papers"):
        """Initialize file manager"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def move_file_to_data_dir(self, source_path: str) -> tuple[bool, str]:
        """Move uploaded file to data directory"""
        try:
            source = Path(source_path)
            destination = self.data_dir / source.name
            
            # If file already exists, add timestamp
            if destination.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = destination.stem, timestamp, destination.suffix
                destination = self.data_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
            
            shutil.move(str(source), str(destination))
            return True, str(destination)
            
        except Exception as e:
            return False, f"Error moving file: {str(e)}"
    
    def get_all_files(self) -> List[Path]:
        """Get all files in data directory"""
        try:
            return list(self.data_dir.glob("*.pdf"))
        except Exception as e:
            print(f"Error getting files: {e}")
            return []
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """Get information about a file"""
        try:
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                return {'error': f'File {filename} not found'}
            
            stat = file_path.stat()
            
            return {
                'filename': filename,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'extension': file_path.suffix,
                'full_path': str(file_path)
            }
            
        except Exception as e:
            return {'error': f'Error getting file info: {str(e)}'}
    
    def delete_file(self, filename: str) -> tuple[bool, str]:
        """Delete a file"""
        try:
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                return False, f"File {filename} not found"
            
            file_path.unlink()
            return True, f"File {filename} deleted successfully"
            
        except Exception as e:
            return False, f"Error deleting file: {str(e)}"


class PerformanceTracker:
    def __init__(self):
        """Initialize performance tracker"""
        self.metrics = {
            'queries_processed': 0,
            'documents_processed': 0,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'last_query_time': 0.0,
            'start_time': time.time()
        }
        self.query_times = []
    
    def record_query(self, processing_time: float):
        """Record a query processing time"""
        self.metrics['queries_processed'] += 1
        self.metrics['last_query_time'] = processing_time
        self.query_times.append(processing_time)
        
        # Calculate average (keep only last 100 queries for efficiency)
        if len(self.query_times) > 100:
            self.query_times = self.query_times[-100:]
        
        self.metrics['average_response_time'] = sum(self.query_times) / len(self.query_times)
    
    def record_document_processing(self, processing_time: float):
        """Record document processing time"""
        self.metrics['documents_processed'] += 1
        self.metrics['total_processing_time'] += processing_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        current_time = time.time()
        uptime = current_time - self.metrics['start_time']
        
        return {
            **self.metrics,
            'uptime_seconds': uptime,
            'uptime_formatted': self._format_duration(uptime)
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'queries_processed': 0,
            'documents_processed': 0,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'last_query_time': 0.0,
            'start_time': time.time()
        }
        self.query_times = []
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"


class SystemManager:
    def __init__(self):
        """Initialize system manager"""
        self.config = {
            'default_search_results': 5,
            'max_chunk_size': 1000,
            'chunk_overlap': 100,
            'max_file_size_mb': 50
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'disk_free_gb': round(psutil.disk_usage('.').free / (1024**3), 2),
                'current_directory': str(Path.cwd()),
                'config': self.config
            }
        except Exception as e:
            return {
                'error': f'Could not get system info: {str(e)}',
                'config': self.config
            }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def save_config(self, file_path: str = "config.json"):
        """Save configuration to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def load_config(self, file_path: str = "config.json"):
        """Load configuration from file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    loaded_config = json.load(f)
                self.config.update(loaded_config)
                return True
        except Exception as e:
            print(f"Error loading config: {e}")
        return False


class ValidationUtils:
    @staticmethod
    def validate_pdf_file(file_path: str) -> tuple[bool, str]:
        """Validate PDF file"""
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            return False, "File must be a PDF"
        
        # Check file size (50MB limit)
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:
            return False, "File size exceeds 50MB limit"
        
        # Try to read first few bytes to check if it's a valid PDF
        try:
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if not header.startswith(b'%PDF-'):
                    return False, "File is not a valid PDF"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
        
        return True, "Valid PDF file"
    
    @staticmethod
    def validate_query(query: str) -> tuple[bool, str]:
        """Validate search query"""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query.strip()) < 3:
            return False, "Query must be at least 3 characters long"
        
        if len(query) > 500:
            return False, "Query is too long (max 500 characters)"
        
        return True, "Valid query"
    
    @staticmethod
    def validate_chunk_parameters(chunk_size: int, overlap: int) -> tuple[bool, str]:
        """Validate chunking parameters"""
        if chunk_size < 100:
            return False, "Chunk size must be at least 100 words"
        
        if chunk_size > 2000:
            return False, "Chunk size cannot exceed 2000 words"
        
        if overlap < 0:
            return False, "Overlap cannot be negative"
        
        if overlap >= chunk_size:
            return False, "Overlap must be smaller than chunk size"
        
        return True, "Valid parameters"


class LoggingUtils:
    @staticmethod
    def setup_logging(log_file: str = "rag_system.log"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def log_query(query: str, response_time: float, num_results: int):
        """Log a query"""
        logging.info(f"Query: '{query[:50]}...' | Time: {response_time:.2f}s | Results: {num_results}")
    
    @staticmethod
    def log_document_processing(filename: str, num_chunks: int, processing_time: float):
        """Log document processing"""
        logging.info(f"Processed: {filename} | Chunks: {num_chunks} | Time: {processing_time:.2f}s")
    
    @staticmethod
    def log_error(error_msg: str, context: str = ""):
        """Log an error"""
        if context:
            logging.error(f"{context}: {error_msg}")
        else:
            logging.error(error_msg)
    
    @staticmethod
    def log_system_event(event: str, details: str = ""):
        """Log a system event"""
        if details:
            logging.info(f"System Event - {event}: {details}")
        else:
            logging.info(f"System Event - {event}")


# Initialize logging when module is imported
LoggingUtils.setup_logging()