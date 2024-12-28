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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


# Usage example
if __name__ == "__main__":
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    indexer = DocumentIndexer(hf_token=HF_TOKEN)
    response = indexer.query_index("Explain briefly what is a Distributed System?")
    # response = indexer.query_index("Who is the president of Kenya?")
    print(response)