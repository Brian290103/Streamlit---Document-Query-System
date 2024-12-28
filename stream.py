import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import ServiceContext, StorageContext
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.core import load_index_from_storage
import os
from huggingface_hub import login
from typing import Optional
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Streamlit state
if 'indexer' not in st.session_state:
    st.session_state.indexer = None
if 'response' not in st.session_state:
    st.session_state.response = None


class DocumentIndexer:
    def __init__(self, hf_token: str, persist_dir: str = "./db"):
        self.hf_token = hf_token
        self.persist_dir = persist_dir
        self._setup_environment()
        self._configure_settings()

    def _setup_environment(self):
        """Set up the environment and authentication."""
        try:
            login(token=self.hf_token)
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = self.hf_token
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face: {e}")
            raise

    def _configure_settings(self):
        """Configure LlamaIndex settings."""
        llm = HuggingFaceInferenceAPI(
            model_name="HuggingFaceH4/zephyr-7b-alpha",
            api_key=self.hf_token
        )

        embed_model = LangchainEmbedding(
            HuggingFaceInferenceAPIEmbeddings(
                api_key=self.hf_token,
                model_name="thenlper/gte-large"
            )
        )

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

    def create_or_load_index(self, data_dir: str = "data") -> VectorStoreIndex:
        """Create a new index or load existing one."""
        if not os.path.exists(self.persist_dir):
            logger.info("Creating new index...")
            return self._create_new_index(data_dir)
        else:
            logger.info("Loading existing index...")
            return self._load_existing_index()

    def _create_new_index(self, data_dir: str) -> VectorStoreIndex:
        """Create a new index from documents."""
        try:
            logger.info("Starting to read documents...")
            documents = SimpleDirectoryReader(data_dir).load_data()
            logger.info(f"Loaded {len(documents)} documents")

            logger.info("Parsing documents into nodes...")
            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} nodes")

            logger.info("Creating vector store index... This may take a few minutes...")
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context
            )

            logger.info("Persisting index to disk...")
            index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info("Index creation completed!")
            return index

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise



    def _load_existing_index(self) -> VectorStoreIndex:
        """Load existing index from storage."""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            return load_index_from_storage(storage_context=storage_context)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def query_index(self, query: str) -> str:
        """Query the index."""
        try:
            index = self.create_or_load_index()
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise



def initialize_indexer():
    """Initialize the DocumentIndexer with HuggingFace token."""
    try:
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            st.error("HuggingFace token not found. Please check your .env file.")
            return None
        return DocumentIndexer(hf_token=hf_token)
    except Exception as e:
        st.error(f"Error initializing indexer: {str(e)}")
        return None


def main():
    # Set up the Streamlit page
    st.set_page_config(
        page_title="Document Query System",
        page_icon="📚",
        layout="wide"
    )

    # Main title
    st.title("📚 Document Query System")
    st.markdown("---")

    # Initialize indexer if not already done
    if st.session_state.indexer is None:
        with st.spinner("Initializing system..."):
            st.session_state.indexer = initialize_indexer()
        if st.session_state.indexer is None:
            st.stop()

    # Sidebar for system information
    with st.sidebar:
        st.header("System Information")
        st.info("""
        This system allows you to:
        - Query your document database
        - Get AI-powered responses
        - Explore your documents
        """)

        st.markdown("---")

        # Add model information
        st.subheader("Model Configuration")
        st.write("- LLM: Zephyr-7b-alpha")
        st.write("- Embeddings: GTE-large")

        # Add system status
        st.markdown("---")
        st.subheader("System Status")
        st.success("✅ System Initialized")
        st.success("✅ Models Loaded")

    # Main query interface
    st.header("🔍 Ask Your Question")

    # Query input
    query = st.text_area(
        "Enter your question",
        height=100,
        placeholder="What would you like to know about your documents?"
    )

    # Query button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Submit Query", type="primary"):
            if query:
                with st.spinner("Processing your question..."):
                    try:
                        response = st.session_state.indexer.query_index(query)
                        logger.info(response)
                        st.session_state.response = response
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")

    # Clear button
    with col2:
        if st.button("Clear"):
            st.session_state.response = None
            query = ""
            st.experimental_rerun()

    # Display response
    if st.session_state.response:
        st.markdown("---")
        st.header("📝 Response")
        st.markdown(st.session_state.response)

        # Feedback buttons
        st.markdown("---")
        st.write("Was this response helpful?")
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            st.button("👍 Yes")
        with col2:
            st.button("👎 No")

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, LlamaIndex, and HuggingFace")


if __name__ == "__main__":
    main()