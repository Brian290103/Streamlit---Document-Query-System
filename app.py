from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import ServiceContext, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.langchain  import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.core import load_index_from_storage
from llama_index.core import Settings


import os
from huggingface_hub import  login


HF_TOKEN = "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"
login(token=HF_TOKEN)
os.environ['HUGGINGFACEHUB_API_TOKEN']=HF_TOKEN

PERSIST_DIR ="./db"
llm=HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    api_key=HF_TOKEN
)
embed_model=LangchainEmbedding(
    HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="thenlper/gte-large"
    )
)
# Configure settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

if not os.path.exists(PERSIST_DIR):
    document=SimpleDirectoryReader("data").load_data()
    parser=SimpleNodeParser()
    nodes=parser.get_nodes_from_documents(document)

    storage_context=StorageContext.from_defaults()
    index=VectorStoreIndex(nodes=nodes,
                           storage_context=storage_context)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index =load_index_from_storage(
        storage_context=storage_context
    )

user_prompt="WHat is a Distributed System?"
query_engine=index.as_query_engine()
response=query_engine.query(user_prompt)
print(response)