import asyncio
from typing import List

from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Context
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.response_synthesizers import CompactAndRefine

# Step 1: Define Events for Each Stage
class RetrieveEvent(StartEvent):
    query: str
    index: VectorStoreIndex

class RerankEvent(StartEvent):
    query: str
    retrieved_nodes: list

class SynthesizeEvent(StartEvent):
    query: str
    reranked_nodes: list

# Step 2: Define the Workflow
class RAGWorkflow(Workflow):
    @step
    async def retrieve(self, ev: RetrieveEvent) -> RerankEvent:
        retriever = ev.index.as_retriever(similarity_top_k=10)
        nodes = retriever.retrieve(ev.query)
        return RerankEvent(query=ev.query, retrieved_nodes=nodes)

    @step
    async def rerank(self, ev: RerankEvent) -> SynthesizeEvent:
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=3,
            llm=OpenAI(model="gpt-4o-mini")
        )
        reranked_nodes = reranker.postprocess_nodes(ev.retrieved_nodes, query_str=ev.query)
        return SynthesizeEvent(query=ev.query, reranked_nodes=reranked_nodes)

    @step
    async def synthesize(self, ev: SynthesizeEvent) -> StopEvent:
        llm = OpenAI(model="gpt-4o-mini")
        synthesizer = CompactAndRefine(llm=llm, streaming=False)
        response = synthesizer.synthesize(ev.query, nodes=ev.reranked_nodes)
        return StopEvent(result=response)

# Step 3: Main Async Entrypoint
async def main():
    # Ingest documents and build index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Initialize workflow
    workflow = RAGWorkflow()

    # User query
    query = "What are the health benefits of regular exercise?"

    # Start the workflow by emitting the first event
    result = await workflow.run(
        event=RetrieveEvent(query=query, index=index)
    )

    print("Final synthesized answer:")
    print(result.result)

if __name__ == "__main__":
    asyncio.run(main())

"""
1. Define Custom Events

- Custom events allow you to pass complex data (like raw files, chunked documents, or retrieval configs) between workflow steps.
"""
from llama_index.core.workflow import StartEvent

class CustomIngestionEvent(StartEvent):
    raw_file_paths: list[str]
    chunk_size: int
    overlap: int

class CustomRetrievalEvent(StartEvent):
    query: str
    retrieval_mode: str  # e.g., "vector", "keyword", "hybrid"
    index: object

"""
2. Implement Custom Ingestion Step

This step can include:
- Loading files from various sources
- Parsing, cleaning, and chunking documents
- Adding metadata or running custom preprocessing
"""
from llama_index.core import Document

@step
async def ingest(self, ev: CustomIngestionEvent) -> CustomRetrievalEvent:
    documents = []
    for file_path in ev.raw_file_paths:
        # Example: Load and chunk documents with overlap
        with open(file_path, 'r') as f:
            text = f.read()
        # Custom chunking logic (simplified)
        chunks = [
            text[i:i+ev.chunk_size]
            for i in range(0, len(text), ev.chunk_size - ev.overlap)
        ]
        for chunk in chunks:
            documents.append(Document(text=chunk, metadata={"source": file_path}))
    # Build or update your index here
    index = VectorStoreIndex.from_documents(documents)
    return CustomRetrievalEvent(query="", retrieval_mode="vector", index=index)

"""
3. Implement Custom Retrieval Step

- This can support multiple retrieval strategies (vector, keyword, hybrid)
"""
@step
async def retrieve(self, ev: CustomRetrievalEvent) -> RerankEvent:
    if ev.retrieval_mode == "vector":
        retriever = ev.index.as_retriever(similarity_top_k=10)
        nodes = retriever.retrieve(ev.query)
    elif ev.retrieval_mode == "keyword":
        # Implement keyword search logic here
        nodes = keyword_search(ev.index, ev.query)
    elif ev.retrieval_mode == "hybrid":
        # Combine vector and keyword results, e.g., Reciprocal Rank Fusion
        nodes = hybrid_search(ev.index, ev.query)
    else:
        raise ValueError("Unknown retrieval mode")
    return RerankEvent(query=ev.query, retrieved_nodes=nodes)

"""
4. Compose the Workflow

- Bring all steps together in a workflow class.
"""
from llama_index.core.workflow import Workflow, step

class CustomRAGWorkflow(Workflow):
    ingest = ingest
    retrieve = retrieve
    # Add rerank and synthesize steps as before

    @step
    async def rerank(self, ev: RerankEvent) -> SynthesizeEvent:
        # ... your rerank logic ...

    @step
    async def synthesize(self, ev: SynthesizeEvent) -> StopEvent:
        # ... your synthesis logic ...

"""
5. Run the workflow
"""
import asyncio

async def main():
    workflow = CustomRAGWorkflow()
    ingestion_event = CustomIngestionEvent(
        raw_file_paths=["data/doc1.txt", "data/doc2.txt"],
        chunk_size=1024,
        overlap=100
    )
    result = await workflow.run(event=ingestion_event)
    print(result.result)

if __name__ == "__main__":
    asyncio.run(main())







