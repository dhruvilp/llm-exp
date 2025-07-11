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
