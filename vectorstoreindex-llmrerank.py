# Install necessary packages if you haven't already
# pip install llama-index-core llama-index-llms-openai

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.indices.query.schema import QueryBundle
from llama_index.postprocessor.llm_rerank import LLMRerank

# 1. Load your data and build the index (adjust directory as needed)
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 2. Set up the service context with your preferred LLM
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4o-mini")  # Use your preferred model
)

# 3. Retrieve initial candidates using embedding-based retrieval
query_str = "What are the health benefits of regular exercise?"
query_bundle = QueryBundle(query_str)
retriever = index.as_retriever(similarity_top_k=10)
retrieved_nodes = retriever.retrieve(query_bundle)

# 4. Rerank the retrieved nodes using LLMRerank
reranker = LLMRerank(
    choice_batch_size=5,  # Number of candidates per LLM call
    top_n=3,              # How many top results to keep
    service_context=service_context
)
reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

# 5. Print reranked results with their scores and text
print("Reranked results:")
for idx, node in enumerate(reranked_nodes):
    print(f"Rank {idx+1}:")
    print(f"Score: {getattr(node, 'score', 'N/A')}")
    print(f"Text: {node.get_content()}\n")
