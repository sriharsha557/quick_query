import streamlit as st
import tempfile
import os
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import base64
import io
import pickle
from dotenv import load_dotenv

# CrewAI and LLM imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Document processing imports
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import numpy as np

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Prefer Streamlit Cloud secrets, fallback to local .env with proper error handling
try:
    groq_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    groq_key = None

# If secrets not found, try .env
if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("⚠ No GROQ_API_KEY found. Please set it in Streamlit Secrets or .env")
    st.info("For Streamlit Cloud: Add GROQ_API_KEY to your app secrets")
    st.info("For local development: Add GROQ_API_KEY to your .env file")
    st.stop()

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
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None
    agent_info: Optional[Dict[str, Any]] = None

class DocumentSearchTool(BaseTool):
    """Custom CrewAI tool for document search"""
    name: str = "document_search"
    description: str = "Search through uploaded documents for relevant information based on a query. Use this tool to find relevant content from uploaded documents before answering questions."
    
    def __init__(self, embeddings_manager=None):
        super().__init__()
        self._embeddings_manager = embeddings_manager
    
    def _run(self, query: str) -> str:
        """Execute the document search"""
        try:
            if not self._embeddings_manager:
                return "Document search system not initialized."
            
            relevant_docs = self._embeddings_manager.similarity_search(query, k=5)
            if not relevant_docs:
                return "No relevant documents found for this query."
            
            # Combine relevant chunks with source information
            results = []
            for doc in relevant_docs:
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                results.append(f"**Source: {source}**\n{content}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the tool"""
        return self._run(query)

class DocumentProcessor:
    """Handles document loading and processing"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
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
        """Extract text from DOCX file"""
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
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error processing TXT: {str(e)}")
            return ""
    
    @classmethod
    def process_uploaded_file(cls, uploaded_file) -> tuple[str, str]:
        """Process uploaded file and return filename and extracted text"""
        # Save uploaded file to temporary location
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
            # Clean up temporary file
            os.unlink(tmp_path)

class EmbeddingsManager:
    """Handles document embeddings and vector storage using FAISS (Streamlit Cloud compatible)"""
    
    def __init__(self):
        self.vectorstore = None
        self.documents_metadata = []  # Store metadata separately
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # Use free HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Try to load existing vectorstore from session state
        self._load_existing_vectorstore()
    
    def _load_existing_vectorstore(self):
        """Try to load existing vectorstore from session state"""
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
        """Create embeddings for documents and store in FAISS vector database"""
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                st.error("No text chunks created from documents")
                return False
            
            # Create or update vector store
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                self.documents_metadata = [doc.metadata for doc in chunks]
            else:
                # Add to existing vectorstore
                new_vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                self.vectorstore.merge_from(new_vectorstore)
                self.documents_metadata.extend([doc.metadata for doc in chunks])
            
            # Store in session state for persistence within session
            st.session_state['faiss_vectorstore'] = self.vectorstore
            st.session_state['documents_metadata'] = self.documents_metadata
            
            return True
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore:
            try:
                # FAISS similarity search returns documents with content and metadata
                results = self.vectorstore.similarity_search(query, k=k)
                return results
            except Exception as e:
                st.error(f"Error in similarity search: {e}")
                return []
        return []
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vectorstore"""
        return len(self.documents_metadata) if self.documents_metadata else 0
    
    def clear_documents(self) -> bool:
        """Clear all stored documents"""
        try:
            self.vectorstore = None
            self.documents_metadata = []
            
            # Clear from session state
            if 'faiss_vectorstore' in st.session_state:
                del st.session_state['faiss_vectorstore']
            if 'documents_metadata' in st.session_state:
                del st.session_state['documents_metadata']
            
            return True
        except Exception as e:
            st.error(f"Error clearing documents: {e}")
            return False

class CrewAIRAGSystem:
    """CrewAI-based RAG system with specialized agents using Groq LLaMA 3"""
    
    def __init__(self, embeddings_manager: EmbeddingsManager):
        self.embeddings_manager = embeddings_manager
        # Use the global groq_key variable
        self.groq_api_key = groq_key
        self.llm = None
        self.agents = {}
        self.tools = []
        
        if self.groq_api_key:
            try:
                # Use Groq's LLaMA 3 with correct model name
                self.llm = ChatGroq(
                    groq_api_key=self.groq_api_key,
                    model="llama-3.1-8b-instant",  # Remove 'groq/' prefix
                    temperature=0.7,
                    max_tokens=1500
                )
                # Test the connection by making a simple call
                test_response = self.llm.invoke("Hello")
                print(f"LLM initialized successfully: {test_response.content[:50]}...")
                self._setup_agents_and_tools()
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                st.error(f"Failed to initialize Groq LLM: {e}")
                self.llm = None
    
    def _setup_agents_and_tools(self):
        """Setup CrewAI agents and tools"""
        if not self.llm:
            return
            
        # Create document search tool
        doc_search_tool = DocumentSearchTool(embeddings_manager=self.embeddings_manager)
        self.tools = [doc_search_tool]
        
        # Data Vault Expert Agent
        self.agents['data_vault_expert'] = Agent(
            role='Data Vault 2.0 Expert',
            goal='Provide accurate and detailed information about Data Vault 2.0 methodology, architecture, and best practices',
            backstory="""You are a senior data architect with extensive experience in Data Vault 2.0 methodology. 
            You understand hub, link, and satellite structures, temporal aspects, and business keys. You can explain 
            complex data vault concepts in clear, actionable terms.""",
            verbose=True,  # Enable verbose for debugging
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
        
        # VaultSpeed Expert Agent
        self.agents['vaultspeed_expert'] = Agent(
            role='VaultSpeed Specialist',
            goal='Provide expert guidance on VaultSpeed automation, configuration, and implementation',
            backstory="""You are a VaultSpeed specialist with deep knowledge of automation tools, 
            code generation, metadata management, and deployment strategies. You help teams implement 
            efficient data vault solutions using VaultSpeed technology.""",
            verbose=True,  # Enable verbose for debugging
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
        
        # Airflow Expert Agent
        self.agents['airflow_expert'] = Agent(
            role='Apache Airflow Expert',
            goal='Provide comprehensive guidance on Airflow orchestration, DAGs, and data pipeline management',
            backstory="""You are an experienced data engineer specializing in Apache Airflow. You understand 
            DAG design patterns, task dependencies, scheduling, monitoring, and integration with data vault systems. 
            You provide practical solutions for complex orchestration challenges.""",
            verbose=True,  # Enable verbose for debugging
            allow_delegation=False,
            tools=self.tools,
            llm=self.llm
        )
        
        # Research Coordinator Agent
        self.agents['coordinator'] = Agent(
            role='Technical Research Coordinator',
            goal='Coordinate research across different domains and provide comprehensive, well-structured responses',
            backstory="""You are a technical coordinator who excels at synthesizing information from multiple 
            expert sources. You ensure responses are comprehensive, well-organized, and address all aspects 
            of complex technical questions.""",
            verbose=True,  # Enable verbose for debugging
            allow_delegation=True,
            tools=self.tools,
            llm=self.llm
        )
    
    # Add this debug version of the generate_response method to your CrewAIRAGSystem class:

    def generate_response(self, query: str, mode: str = "Overview") -> tuple[str, List[str], Dict[str, Any]]:
        """Generate response using CrewAI agents"""
        # Simple test first
        if not self.llm:
            return "ERROR: No LLM found", [], {}
    
        if not self.agents:
            return "ERROR: No agents found", [], {}
    
        try:
            # Just test the LLM directly first
            simple_response = self.llm.invoke(f"Answer this question about data vault: {query}")
        
            return f"✅ SUCCESS! LLM Response: {simple_response.content}", [], {
                'primary_agent': 'test',
                'agent_role': 'Direct LLM Test',
                'mode': mode
            }
        
        except Exception as e:
            return f"ERROR in LLM call: {str(e)}", [], {}
    
    # Add debug information
        st.write(f"🔍 **Debug Checks:**")
        st.write(f"- LLM status: {self.llm is not None}")
        st.write(f"- LLM type: {type(self.llm)}")
        st.write(f"- Number of agents: {len(self.agents)}")
        st.write(f"- Available agents: {list(self.agents.keys())}")
        st.write(f"- API Key present: {bool(self.groq_api_key)}")
    
    # Check the condition that determines fallback
        llm_check = self.llm is not None
        agents_check = bool(self.agents)
    
        st.write(f"- LLM check result: {llm_check}")
        st.write(f"- Agents check result: {agents_check}")
        st.write(f"- Combined check (should be True): {llm_check and agents_check}")
    
        if not self.llm or not self.agents:
            st.error("🚨 **Debug: Entering fallback mode**")
            st.write(f"Reason: LLM={self.llm is not None}, Agents={bool(self.agents)}")
            return self._fallback_response(query, mode), [], {}
    
        st.success("✅ **Debug: Conditions met, proceeding with CrewAI**")
    
        try:
        # Determine the best agent for this query
            primary_agent = self._determine_relevant_agent(query)
            st.write(f"🤖 **Debug: Selected agent: {primary_agent}**")
        
        # Check if the selected agent exists
            if primary_agent not in self.agents:
                st.error(f"❌ Selected agent '{primary_agent}' not found in agents!")
                return self._fallback_response(query, mode), [], {}
        
            st.write(f"✅ Agent '{primary_agent}' found successfully")
        
        # Create task based on response mode
            if mode == "Overview":
                task_description = f"""Analyze the user query: "{query}"
            
                First, use the document_search tool to find relevant information from the uploaded documents.
                Then provide a concise overview response that includes:
                1. Key points relevant to the query (use bullet points)
                2. Brief explanations of important concepts
                3. Practical recommendations if applicable
            
                Keep the response focused and easy to understand."""
            else:  # Deep Dive
                task_description = f"""Analyze the user query: "{query}"
            
                First, use the document_search tool to find comprehensive information from the uploaded documents.
                Then provide a detailed deep-dive response that includes:
                1. Detailed explanation of relevant concepts
                2. Technical implementation details
                3. Best practices and recommendations
                4. Potential challenges and solutions
                5. Integration considerations if applicable
            
                Provide thorough, technical depth while maintaining clarity."""
        
            st.write("📝 **Debug: Creating task...**")
        
            # Create and execute task
            task = Task(
            description=task_description,
            agent=self.agents[primary_agent],
            expected_output="A well-structured response based on the document content and agent expertise"
            )
            st.write("👥 **Debug: Creating crew...**")
        
            # Create crew and execute
            crew = Crew(
            agents=[self.agents[primary_agent]],
            tasks=[task],
            verbose=True,  # Enable verbose for debugging
            process=Process.sequential
            )
        
            st.write("🚀 **Debug: Executing CrewAI task...**")
        
            with st.spinner("CrewAI agents processing..."):
                result = crew.kickoff()
        
            st.write(f"✅ **Debug: Task completed successfully!**")
            st.write(f"Result type: {type(result)}")
            st.write(f"Result preview: {str(result)[:200]}...")
        
            # Extract sources from the search results
            sources = self._extract_sources_from_search()
        
            agent_info = {
            'primary_agent': primary_agent,
            'agent_role': self.agents[primary_agent].role,
            'mode': mode
            }
        
            return str(result), sources, agent_info
        
        except Exception as e:
            st.error(f"💥 **Debug: CrewAI execution error occurred**")
            st.error(f"Error: {str(e)}")
            st.error(f"Error type: {type(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return self._fallback_response(query, mode), [], {}
    
    def _extract_sources_from_search(self) -> List[str]:
        """Extract sources from recent search operations"""
        # This would be enhanced to track sources from tool usage
        # For now, return a placeholder
        if hasattr(st.session_state, 'recent_sources'):
            return st.session_state.recent_sources
        return []
    
    def _fallback_response(self, query: str, mode: str) -> str:
        """Fallback response when CrewAI is not available"""
        if mode == "Overview":
            return f"""**Overview Response for: "{query}"**

• **Configuration Required**: CrewAI system requires Groq API key in .env file
• **Fallback Mode**: Currently operating in basic mode
• **Expected Features**: With proper configuration, you'll get:
  - Expert agent analysis
  - Document-based responses
  - Specialized domain knowledge

*Please ensure your .env file contains: GROQ_API_KEY=your_key_here*"""
        else:
            return f"""**Deep Dive Analysis - Configuration Required**

Your query "{query}" would normally be processed by our specialized CrewAI agents using Groq's LLaMA 3:

**Data Vault Expert**: For questions about DV2.0 methodology, hubs, links, satellites
**VaultSpeed Specialist**: For automation, code generation, and tooling questions  
**Airflow Expert**: For orchestration, DAGs, and pipeline management

**To enable full functionality:**
1. Create a .env file in your project root
2. Add your Groq API key: GROQ_API_KEY=your_key_here
3. Restart the application

**Current Status**: Operating in fallback mode without agent-based analysis."""

class RAGChatbot:
    """Main RAG chatbot class with CrewAI integration"""
    
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        self.crewai_system = CrewAIRAGSystem(self.embeddings_manager)
    
    def load_documents(self, uploaded_files) -> bool:
        """Load and process uploaded documents"""
        if not uploaded_files:
            return False
        
        documents = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            filename, text = DocumentProcessor.process_uploaded_file(uploaded_file)
            
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": filename, "upload_time": datetime.now().isoformat()}
                )
                documents.append(doc)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if documents:
            success = self.embeddings_manager.create_embeddings(documents)
            if success:
                st.session_state.documents_loaded = True
                st.session_state.document_count = self.embeddings_manager.get_document_count()
                # Store sources for reference
                st.session_state.recent_sources = [doc.metadata['source'] for doc in documents]
                return True
        
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
        """Generate response using CrewAI RAG system"""
        result = self.crewai_system.generate_response(query, mode)
        return result

def load_image_as_base64(image_path: str) -> str:
    """Load image and convert to base64 - supports both local and git paths"""
    # Define possible paths
    local_path = f"D:\\MOOD\\CODE\\{image_path}"
    git_path = image_path
    
    # Try local path first, then git path
    paths_to_try = [local_path, git_path]
    
    for path in paths_to_try:
        try:
            if os.path.exists(path):
                with open(path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            continue
    
    # If no paths work, show warning but don't error
    st.warning(f"Image not found at either: {local_path} or {git_path}")
    return ""

def render_sidebar():
    """Render the sidebar with controls"""
    with st.sidebar:
        # Load book.png for sidebar logo
        book_img_path = "images/Quickquery.png"
        book_b64 = load_image_as_base64(book_img_path)
        
        if book_b64:
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{book_b64}" 
                        style="width: 140px; height: auto; margin-bottom: 10px;" />
                    <p style="margin: 0; font-style: italic; color: #666; font-size: 14px;">
                        Powered by CrewAI & Groq
                    </p>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("# 📚 Quick Query")
            st.markdown("*Powered by CrewAI & Groq*")
        
        st.markdown("---")
        
        # Environment status
        groq_key_status = "✅ Connected" if groq_key else "❌ Not Found"
        st.markdown(f"**Groq LLaMA 3:** {groq_key_status}")
        
        if not groq_key:
            st.warning("⚠️ Add GROQ_API_KEY to Streamlit Secrets or .env file for full functionality")
        
        st.markdown("---")
        
        # File uploader
        st.markdown("### 📁 Document Management")
        
        # Show current document status
        if st.session_state.get('documents_loaded', False):
            doc_count = st.session_state.get('document_count', 0)
            st.success(f"✅ {doc_count} document chunks loaded (session only)")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All Documents"):
                    if st.session_state.chatbot.clear_all_documents():
                        st.success("Documents cleared!")
                        st.rerun()
                    else:
                        st.error("Failed to clear documents")
            
            with col2:
                if st.button("📤 Add More Documents"):
                    st.session_state.show_uploader = True
                    st.rerun()
        
        # Show uploader if no documents or user wants to add more
        if not st.session_state.get('documents_loaded', False) or st.session_state.get('show_uploader', False):
            uploaded_files = st.file_uploader(
                "Choose files to add",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload PDF, DOCX, or TXT files. Documents persist for current session only.",
                key="file_uploader"
            )
            
            if uploaded_files:
                if st.button("📤 Upload & Process"):
                    with st.spinner("Processing and storing documents..."):
                        success = st.session_state.chatbot.load_documents(uploaded_files)
                        if success:
                            st.success("Documents processed and stored!")
                            st.session_state.show_uploader = False
                            st.rerun()
                        else:
                            st.error("Failed to process documents")
        else:
            uploaded_files = None
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        # Response mode
        response_mode = st.selectbox(
            "Response Mode",
            ["Overview", "Deep Dive"],
            help="Overview: Concise bullet points | Deep Dive: Detailed analysis"
        )
        
        # Show sources toggle
        show_sources = st.toggle(
            "Show Sources",
            value=True,
            help="Display source documents for each answer"
        )
        
        # Show agent info toggle
        show_agent_info = st.toggle(
            "Show Agent Info",
            value=True,
            help="Display which CrewAI agent handled the query"
        )
        
        # Document status
        if st.session_state.get('documents_loaded', False):
            st.info(f"📄 {st.session_state.get('document_count', 0)} chunks loaded")
            st.caption("⚠️ Documents reset on page refresh")
        
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
                # Agent info header
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
    
    # Custom CSS with plain white header
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
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    uploaded_files, response_mode, show_sources, show_agent_info = render_sidebar()
    
    # Main content area - Header with Quickquery.png
    quickquery_img_path = "images/Quickquery.png"
    quickquery_b64 = load_image_as_base64(quickquery_img_path)
    
    if quickquery_b64:
        st.markdown(f'''
            <div class="main-header" style="margin-bottom: 20px;">
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
    
    # --- Topics section ---
    st.markdown("---")
    st.markdown("### 📚 List of Topics Available for Search")
    st.markdown("""
    - Data Vault 2.0 Fundamentals  
    - Hubs, Links, and Satellites  
    - Business Keys and Surrogate Keys  
    - Staging Layer Best Practices  
    - PIT & Bridge Tables  
    - Data Vault vs Dimensional Modeling  
    - Automation in Data Vault  
    - Agile Delivery with Data Vault  
    """)
    
    # Handle file uploads only if new files are uploaded
    if uploaded_files and not st.session_state.get('vectorstore_loaded', False):
        # This will only run for new uploads, not on every rerun
        pass
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            render_chat_message(message, show_sources, show_agent_info)
    
    # Input area
    with st.container():
        if st.session_state.documents_loaded:
            # Chat input
            user_input = st.chat_input("Ask a question about your documents...")
            
            if user_input:
                # Add user message to history
                user_message = ChatMessage(
                    role="user",
                    content=user_input,
                    timestamp=datetime.now()
                )
                st.session_state.chat_history.append(user_message)
                
                # Generate response
                with st.spinner("CrewAI agents are analyzing your question using Groq LLaMA 3..."):
                    response_content, sources, agent_info = st.session_state.chatbot.generate_response(
                        user_input, response_mode
                    )
                
                # Add assistant message to history
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
            st.info("👆 Please upload documents using the sidebar to start chatting with CrewAI agents!")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()

