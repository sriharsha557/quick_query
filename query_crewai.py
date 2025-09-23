import streamlit as st
import tempfile
import os
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    import sqlite3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import base64
import io
import pickle
from dotenv import load_dotenv

# Document processing imports
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import numpy as np

# LLM import - THIS WAS MISSING!
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Get API key with better error handling
groq_key = None
try:
    groq_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    pass

if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("No GROQ_API_KEY found. Please set it in Streamlit Secrets or .env")
    # Don't use st.stop() - let the app continue with limited functionality

# Configure page
st.set_page_config(
    page_title="Quick Query",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

@dataclass
class ChatMessage:
    """Data class for chat messages"""
    role: str
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None
    agent_info: Optional[Dict[str, Any]] = None

# Simplified Document Search Tool with better error handling
class SimpleDocumentSearchTool:
    """Simplified document search without BaseTool inheritance"""
    
    def __init__(self, embeddings_manager=None):
        self.embeddings_manager = embeddings_manager
        self.name = "document_search"
    
    def search(self, query: str) -> str:
        """Execute the document search with relevance assessment"""
        try:
            if not self.embeddings_manager:
                return "SEARCH_ERROR: Document search system not initialized."
            
            relevant_docs = self.embeddings_manager.similarity_search(query, k=5)
            if not relevant_docs:
                return "NO_CONTENT_FOUND: No relevant documents found for this query."
            
            results = []
            total_content_length = 0
            
            for i, doc in enumerate(relevant_docs):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content.strip()
                total_content_length += len(content)
                
                if len(content) > 800:
                    content = content[:800] + "..."
                
                results.append(f"**Source {i+1}: {source}**\n{content}")
            
            if total_content_length < 100:
                relevance_note = "\n\nRELEVANCE_LOW: Limited content found."
            elif total_content_length < 500:
                relevance_note = "\n\nRELEVANCE_MEDIUM: Some relevant content found."
            else:
                relevance_note = "\n\nRELEVANCE_HIGH: Substantial relevant content found."
            
            return "\n\n".join(results) + relevance_note
            
        except Exception as e:
            return f"SEARCH_ERROR: Error searching documents: {str(e)}"

class DocumentProcessor:
    """Handles document loading and processing"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error processing DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error processing TXT: {str(e)}")
            return ""
    
    @classmethod
    def process_uploaded_file(cls, uploaded_file) -> tuple[str, str]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        filename = uploaded_file.name
        file_ext = filename.lower().split('.')[-1]
        
        try:
            if file_ext == 'pdf':
                text = cls.extract_text_from_pdf(tmp_path)
            elif file_ext == 'docx':
                text = cls.extract_text_from_docx(tmp_path)
            elif file_ext == 'txt':
                text = cls.extract_text_from_txt(tmp_path)
            else:
                st.error(f"Unsupported file type: {file_ext}")
                text = ""
            
            return filename, text
        finally:
            os.unlink(tmp_path)

class EmbeddingsManager:
    """Handles document embeddings and vector storage"""
    
    def __init__(self):
        self.vectorstore = None
        self.documents_metadata = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except ImportError as e:
                st.error(f"Could not import HuggingFaceEmbeddings: {e}")
                self.embeddings = None
        
        self._load_existing_vectorstore()
    
    def _load_existing_vectorstore(self):
        try:
            if 'faiss_vectorstore' in st.session_state and 'documents_metadata' in st.session_state:
                self.vectorstore = st.session_state['faiss_vectorstore']
                self.documents_metadata = st.session_state['documents_metadata']
                if self.vectorstore is not None:
                    st.session_state.documents_loaded = True
                    st.session_state.document_count = len(self.documents_metadata)
                    st.session_state.vectorstore_loaded = True
        except Exception as e:
            st.warning(f"Could not load existing documents: {e}")
    
    def create_embeddings(self, documents: List[Document]) -> bool:
        try:
            if not self.embeddings:
                st.error("Embeddings system not initialized")
                return False
                
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                st.error("No text chunks created from documents")
                return False
            
            MAX_CHUNKS = 1000
            if len(chunks) > MAX_CHUNKS:
                st.warning(f"Processing first {MAX_CHUNKS} chunks for optimal performance.")
                chunks = chunks[:MAX_CHUNKS]
            
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                self.documents_metadata = [doc.metadata for doc in chunks]
            else:
                new_vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                self.vectorstore.merge_from(new_vectorstore)
                self.documents_metadata.extend([doc.metadata for doc in chunks])
            
            st.session_state['faiss_vectorstore'] = self.vectorstore
            st.session_state['documents_metadata'] = self.documents_metadata
            
            return True
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if self.vectorstore:
            try:
                results = self.vectorstore.similarity_search(query, k=k)
                return results
            except Exception as e:
                st.error(f"Error in similarity search: {e}")
                return []
        return []
    
    def get_document_count(self) -> int:
        return len(self.documents_metadata) if self.documents_metadata else 0
    
    def clear_documents(self) -> bool:
        try:
            self.vectorstore = None
            self.documents_metadata = []
            
            if 'faiss_vectorstore' in st.session_state:
                del st.session_state['faiss_vectorstore']
            if 'documents_metadata' in st.session_state:
                del st.session_state['documents_metadata']
            
            return True
        except Exception as e:
            st.error(f"Error clearing documents: {e}")
            return False

class SimplifiedRAGSystem:
    """Simplified RAG system without CrewAI complexity"""
    
    def __init__(self, embeddings_manager: EmbeddingsManager):
        self.embeddings_manager = embeddings_manager
        self.groq_api_key = groq_key
        self.llm = None
        self.doc_search_tool = None
        self.initialization_error = None
        
        try:
            # Initialize LLM with better error handling
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=1500,
                timeout=30,  # Add timeout
                max_retries=2  # Add retries
            )
            
            # Test LLM connection with a simple query
            test_response = self.llm.invoke("Hello")
            if hasattr(test_response, 'content'):
                st.sidebar.success("✅ LLM connected successfully")
            else:
                raise Exception("Invalid response format from LLM")
            
            # Initialize document search
            self.doc_search_tool = SimpleDocumentSearchTool(embeddings_manager=self.embeddings_manager)
            
        except Exception as e:
            self.initialization_error = str(e)
            st.sidebar.error(f"❌ LLM initialization failed: {str(e)}")
            self.llm = None
    
    def generate_response(self, query: str, mode: str = "Overview") -> tuple[str, List[str], Dict[str, Any]]:
        """Generate response using simplified approach"""
        
        if not self.llm:
            return self._fallback_response(query, mode), [], {}
        
        try:
            # Step 1: Search documents
            search_results = self.doc_search_tool.search(query)
            
            # Step 2: Determine if we have sufficient content
            has_sufficient_content = (
                "RELEVANCE_HIGH" in search_results or 
                "RELEVANCE_MEDIUM" in search_results
            ) and not (
                "NO_CONTENT_FOUND" in search_results or 
                "SEARCH_ERROR" in search_results
            )
            
            # Step 3: Generate response based on content availability
            if has_sufficient_content:
                # Use document content
                if mode == "Overview":
                    prompt = f"""Based on the following document content, provide a concise overview answering the user's query: "{query}"

Document Content:
{search_results}

Instructions:
- Provide a clear, concise answer based on the document content
- Cite the sources mentioned in the document content
- If multiple sources are mentioned, reference them appropriately
- Keep the response focused and informative
- Use bullet points or numbered lists when appropriate for clarity

Query: {query}
"""
                else:  # Deep Dive
                    prompt = f"""Based on the following document content, provide a detailed analysis answering the user's query: "{query}"

Document Content:
{search_results}

Instructions:
- Provide a comprehensive, detailed answer based on the document content
- Include technical details and specifics found in the documents
- Cross-reference information from different sources if applicable
- Cite all sources mentioned in the document content
- Provide implementation guidance if present in documents
- Structure your response with clear headings and sections
- Use examples from the documents when available

Query: {query}
"""
                
                response = self.llm.invoke(prompt)
                
                # Extract sources
                sources = self._extract_sources_from_search_results(search_results)
                
                agent_info = {
                    'primary_agent': 'document_analyzer',
                    'agent_role': 'Document Content Analyzer',
                    'mode': mode,
                    'search_quality': 'sufficient_content'
                }
                
                return response.content, sources, agent_info
            
            else:
                # Insufficient content - ask for confirmation
                insufficient_response = f"I couldn't find sufficient information about '{query}' in your uploaded documents. Would you like me to provide a general answer based on my knowledge instead?"
                
                agent_info = {
                    'primary_agent': 'document_analyzer',
                    'agent_role': 'Document Content Analyzer',
                    'mode': mode,
                    'search_quality': 'insufficient_content'
                }
                
                return insufficient_response, [], agent_info
        
        except Exception as e:
            st.sidebar.error(f"💥 Response generation error: {str(e)}")
            return self._fallback_response(query, mode), [], {}
    
    def generate_llm_response(self, query: str, mode: str = "Overview") -> tuple[str, List[str], Dict[str, Any]]:
        """Generate response using general knowledge"""
        if not self.llm:
            return "LLM not available", [], {}
        
        try:
            if mode == "Overview":
                prompt = f"""Provide a helpful, concise answer to the following question using your general knowledge:

Question: {query}

Instructions:
- Provide a clear, informative overview
- Be accurate and helpful
- Mention that this answer is based on general knowledge, not the uploaded documents
- Keep the response focused and well-structured
- Use bullet points when appropriate for clarity
"""
            else:  # Deep Dive
                prompt = f"""Provide a comprehensive, detailed answer to the following question using your general knowledge:

Question: {query}

Instructions:
- Provide an in-depth, detailed analysis
- Include technical details, examples, and explanations where appropriate
- Mention that this answer is based on general knowledge, not the uploaded documents
- Structure the response clearly with good organization
- Be thorough and informative
- Use headings and sections for better readability
"""
            
            response = self.llm.invoke(prompt)
            
            agent_info = {
                'primary_agent': 'general_knowledge',
                'agent_role': 'General Knowledge Assistant',
                'mode': mode,
                'source': 'general_knowledge'
            }
            
            return response.content, [], agent_info
            
        except Exception as e:
            st.error(f"Error in LLM response: {str(e)}")
            return self._fallback_response(query, mode), [], {}
    
    def _extract_sources_from_search_results(self, search_results: str) -> List[str]:
        """Extract unique document sources, not individual chunks"""
        sources = set()  # Use set to avoid duplicates
        lines = search_results.split('\n')
        for line in lines:
            if line.startswith('**Source') and ':' in line:
                source_part = line.split(':', 1)[1].split('**')[0].strip()
                # Extract just the filename, not the section
                if source_part:
                    # Get just the main filename before any parentheses
                    main_source = source_part.split('(')[0].strip()
                    sources.add(main_source)
        return list(sources)
    
    def _fallback_response(self, query: str, mode: str) -> str:
        """Fallback response when system is not available"""
        return f"""**System Error - Unable to Process Query**

Query: "{query}"
Mode: {mode}

**Issue**: The RAG system is not properly initialized.

**Possible Solutions:**
1. Check your Groq API key configuration
2. Ensure internet connectivity for model access
3. Restart the application
4. Verify all required packages are installed

**Error Details**: {self.initialization_error or 'Unknown initialization error'}

**Next Steps**: Please check the sidebar for system status and try refreshing the page."""

class RAGChatbot:
    """Main RAG chatbot class with simplified system"""
    
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        self.rag_system = SimplifiedRAGSystem(self.embeddings_manager)
    
    def load_documents(self, uploaded_files) -> bool:
        """Load and process uploaded documents"""
        if not uploaded_files:
            return False
        
        documents = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            st.write(f"Processing: {uploaded_file.name}")
            filename, text = DocumentProcessor.process_uploaded_file(uploaded_file)
            
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": filename, "upload_time": datetime.now().isoformat()}
                )
                documents.append(doc)
            else:
                st.warning(f"No text extracted from {filename}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if documents:
            success = self.embeddings_manager.create_embeddings(documents)
            if success:
                st.session_state.documents_loaded = True
                st.session_state.document_count = self.embeddings_manager.get_document_count()
                st.session_state.recent_sources = [doc.metadata['source'] for doc in documents]
                st.success(f"✅ Successfully processed {len(documents)} documents!")
                return True
        
        st.error("❌ No documents were successfully processed")
        return False
    
    def clear_all_documents(self) -> bool:
        """Clear all stored documents"""
        success = self.embeddings_manager.clear_documents()
        if success:
            st.session_state.documents_loaded = False
            st.session_state.document_count = 0
            st.session_state.recent_sources = []
            st.session_state.vectorstore_loaded = False
        return success
    
    def generate_response(self, query: str, mode: str = "Overview") -> tuple[str, List[str], Dict[str, Any]]:
        """Generate response using simplified RAG system"""
        return self.rag_system.generate_response(query, mode)
    
    def generate_llm_response(self, query: str, mode: str = "Overview") -> tuple[str, List[str], Dict[str, Any]]:
        """Generate response using LLM fallback"""
        return self.rag_system.generate_llm_response(query, mode)

# Keep the rest of your existing functions (load_image_as_base64, render_sidebar, render_chat_message, main)
def load_image_as_base64(image_path: str) -> str:
    """Load image and convert to base64"""
    local_path = f"D:\\MOOD\\CODE\\{image_path}"
    git_path = image_path
    
    paths_to_try = [local_path, git_path]
    
    for path in paths_to_try:
        try:
            if os.path.exists(path):
                with open(path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            continue
    
    # Don't show warning for missing images in production
    return ""

def render_sidebar():
    """Render the sidebar with controls"""
    with st.sidebar:
        book_img_path = "images/Quickquery.png"
        book_b64 = load_image_as_base64(book_img_path)
        
        if book_b64:
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{book_b64}" 
                        style="width: 140px; height: auto; margin-bottom: 10px;" />
                    <p style="margin: 0; font-style: italic; color: #666; font-size: 14px;">
                        Simplified RAG System
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("# 📚 Quick Query")
            st.markdown("*Simplified RAG System*")
        
        st.markdown("---")
        
        # Environment status
        groq_key_status = "✅ Connected" if groq_key else "❌ Not Found"
        st.markdown(f"**Groq LLaMA 3:** {groq_key_status}")
        
        if not groq_key:
            st.warning("⚠️ Add GROQ_API_KEY to Streamlit Secrets or .env file")
        
        st.markdown("---")
        
        # Document management
        st.markdown("### 📄 Document Management")
        
        if st.session_state.get('documents_loaded', False):
            doc_count = st.session_state.get('document_count', 0)
            st.success(f"✅ {doc_count} document chunks loaded")
            
            # Show recent sources
            recent_sources = st.session_state.get('recent_sources', [])
            if recent_sources:
                st.markdown("**Recent uploads:**")
                for source in recent_sources[:3]:  # Show max 3
                    st.markdown(f"• {source}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All"):
                    if st.session_state.chatbot.clear_all_documents():
                        st.success("Documents cleared!")
                        st.rerun()
                    else:
                        st.error("Failed to clear documents")
            
            with col2:
                if st.button("📤 Add More"):
                    st.session_state.show_uploader = True
                    st.rerun()
        
        if not st.session_state.get('documents_loaded', False) or st.session_state.get('show_uploader', False):
            uploaded_files = st.file_uploader(
                "Choose files to add",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload PDF, DOCX, or TXT files.",
                key="file_uploader"
            )
            
            if uploaded_files:
                if st.button("📤 Upload & Process"):
                    with st.spinner("Processing documents..."):
                        success = st.session_state.chatbot.load_documents(uploaded_files)
                        if success:
                            st.session_state.show_uploader = False
                            st.rerun()
        else:
            uploaded_files = None
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        response_mode = st.selectbox(
            "Response Mode",
            ["Overview", "Deep Dive"],
            help="Overview: Concise responses | Deep Dive: Detailed analysis"
        )
        
        show_sources = st.toggle(
            "Show Sources",
            value=True,
            help="Display source documents for each answer"
        )
        
        show_agent_info = st.toggle(
            "Show Agent Info",
            value=True,
            help="Display system information"
        )
        
        # Performance info
        if st.session_state.get('documents_loaded', False):
            doc_count = st.session_state.get('document_count', 0)
            st.info(f"📄 {doc_count} chunks loaded")
        
        return uploaded_files, response_mode, show_sources, show_agent_info

def render_chat_message(message: ChatMessage, show_sources: bool = True, show_agent_info: bool = True):
    """Render a single chat message"""
    if message.role == "user":
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col2:
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; text-align: right;">
                    <strong>You:</strong><br>
                    {message.content}
                </div>
                """, unsafe_allow_html=True)
    else:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                agent_header = ""
                if show_agent_info and message.agent_info:
                    agent_role = message.agent_info.get('agent_role', 'Quick Query')
                    mode = message.agent_info.get('mode', '')
                    agent_header = f"<small><strong>🤖 {agent_role}</strong> • {mode} Mode</small><br>"
                
                st.markdown(f"""
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;">
                    {agent_header}
                    <strong>Quick Query:</strong><br>
                    {message.content}
                </div>
                """, unsafe_allow_html=True)
                
                if show_sources and message.sources:
                    st.markdown("**📚 Sources:**")
                    for source in message.sources:
                        st.markdown(f"• {source}")

def main():
    """Main application function"""
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'awaiting_llm_confirmation' not in st.session_state:
        st.session_state.awaiting_llm_confirmation = False
    if 'pending_query' not in st.session_state:
        st.session_state.pending_query = ""
    if 'pending_mode' not in st.session_state:
        st.session_state.pending_mode = ""
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: left;
        padding: 20px;
        margin-bottom: 30px;
        background: white;
        color: #333;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .llm-confirmation {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton > button {
        border-radius: 5px;
        border: 1px solid #ddd;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    uploaded_files, response_mode, show_sources, show_agent_info = render_sidebar()
    
    # Main header
    quickquery_img_path = "images/Quickquery.png"
    quickquery_b64 = load_image_as_base64(quickquery_img_path)
    
    if quickquery_b64:
        st.markdown(f'''
            <div class="main-header">
                <div style="display: flex; align-items: flex-end;">
                    <img src="data:image/png;base64,{quickquery_b64}" 
                        style="width: 240px; height: auto; margin-right: 20px;" />
                    <h1 style="margin: 0; font-size: 28px; font-weight: bold;">
                        Find Answers Inside Your Documents
                    </h1>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="main-header">
            <h1>📚 Find Answers Inside Your Documents</h1>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📚 How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1: Upload**  \n📄 Upload documents (PDF, DOCX, TXT)")
    with col2:
        st.markdown("**Step 2: Ask**  \n❓ Ask questions about your content")
    with col3:
        st.markdown("**Step 3: Get Answers**  \n💡 Get answers from documents or general knowledge")
    
    st.markdown("---")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            render_chat_message(message, show_sources, show_agent_info)
    
    # Handle LLM confirmation
    if st.session_state.awaiting_llm_confirmation:
        st.markdown("""
        <div class="llm-confirmation">
            <h4>🤔 Would you like a general answer?</h4>
            <p>I couldn't find sufficient information in your documents. Would you like me to provide an answer based on my general knowledge?</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Yes, please"):
                with st.spinner("Generating answer..."):
                    response_content, sources, agent_info = st.session_state.chatbot.generate_llm_response(
                        st.session_state.pending_query, st.session_state.pending_mode
                    )
                
                assistant_message = ChatMessage(
                    role="assistant",
                    content=response_content,
                    timestamp=datetime.now(),
                    sources=sources if show_sources else None,
                    agent_info=agent_info
                )
                st.session_state.chat_history.append(assistant_message)
                
                st.session_state.awaiting_llm_confirmation = False
                st.session_state.pending_query = ""
                st.session_state.pending_mode = ""
                st.rerun()
        
        with col2:
            if st.button("❌ No, thanks"):
                assistant_message = ChatMessage(
                    role="assistant",
                    content="No problem! Feel free to ask another question or upload more specific documents that might contain the information you're looking for.",
                    timestamp=datetime.now(),
                    sources=None,
                    agent_info={"agent_role": "System", "mode": ""}
                )
                st.session_state.chat_history.append(assistant_message)
                
                st.session_state.awaiting_llm_confirmation = False
                st.session_state.pending_query = ""
                st.session_state.pending_mode = ""
                st.rerun()
    
    # Input area
    with st.container():
        if st.session_state.documents_loaded:
            if not st.session_state.awaiting_llm_confirmation:
                user_input = st.chat_input("Ask a question about your documents...")
                
                if user_input:
                    user_message = ChatMessage(
                        role="user",
                        content=user_input,
                        timestamp=datetime.now()
                    )
                    st.session_state.chat_history.append(user_message)
                    
                    with st.spinner("Analyzing your documents..."):
                        response_content, sources, agent_info = st.session_state.chatbot.generate_response(
                            user_input, response_mode
                        )
                    
                    if ("couldn't find sufficient information" in response_content.lower() or 
                        "would you like me to provide a general answer" in response_content.lower()):
                        
                        assistant_message = ChatMessage(
                            role="assistant",
                            content=response_content,
                            timestamp=datetime.now(),
                            sources=sources if show_sources else None,
                            agent_info=agent_info
                        )
                        st.session_state.chat_history.append(assistant_message)
                        
                        st.session_state.awaiting_llm_confirmation = True
                        st.session_state.pending_query = user_input
                        st.session_state.pending_mode = response_mode
                        
                    else:
                        assistant_message = ChatMessage(
                            role="assistant",
                            content=response_content,
                            timestamp=datetime.now(),
                            sources=sources if show_sources else None,
                            agent_info=agent_info
                        )
                        st.session_state.chat_history.append(assistant_message)
                    
                    st.rerun()
        else:
            st.info("👆 Please upload documents using the sidebar to start chatting!")
            
            # Show example questions when no documents are loaded
            st.markdown("### 💡 Example Questions You Can Ask:")
            example_questions = [
                "What are the main topics covered in this document?",
                "Can you summarize the key findings?",
                "What recommendations are mentioned?",
                "Are there any specific dates or numbers mentioned?",
                "What is the conclusion of this document?"
            ]
            
            for i, question in enumerate(example_questions, 1):
                st.markdown(f"{i}. *{question}*")
    
    # Footer with clear chat and statistics
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**💬 Chat History:** {len(st.session_state.chat_history)} messages")
        
        with col2:
            if st.button("🔄 New Chat"):
                st.session_state.chat_history = []
                st.session_state.awaiting_llm_confirmation = False
                st.session_state.pending_query = ""
                st.session_state.pending_mode = ""
                st.rerun()
        
        with col3:
            # Export chat option
            if st.button("📥 Export Chat"):
                chat_export = []
                for msg in st.session_state.chat_history:
                    chat_export.append({
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "sources": msg.sources
                    })
                
                import json
                chat_json = json.dumps(chat_export, indent=2)
                st.download_button(
                    label="Download Chat History",
                    data=chat_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

