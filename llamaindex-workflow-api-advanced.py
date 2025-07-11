"""
`pip install llama-index-core llama-index-llms-openai llama-index-utils-workflow aiohttp pillow`

Additional resources:
1. https://docs.llamaindex.ai/en/stable/examples/workflow/workflows_cookbook/
2. https://docs.llamaindex.ai/en/stable/examples/workflow/rag/

"""

import asyncio
import aiohttp
import logging
from typing import List, Optional, Union
from PIL import Image
from llama_index.core.workflow import (
    Workflow, StartEvent, StopEvent, Event, step, Context
)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.utils.workflow import draw_all_possible_flows, draw_most_recent_execution
from llama_index.core.workflow.checkpointer import WorkflowCheckpointer

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedRAGWorkflow")

# --- Custom Events ---
class IngestionEvent(StartEvent):
    filepaths: List[str]
    imagepaths: Optional[List[str]] = None
    chunk_size: int = 1024
    overlap: int = 100

class RetrievalEvent(Event):
    query: str
    index: VectorStoreIndex
    retrieval_mode: str = "hybrid"

class RerankEvent(Event):
    query: str
    retrieved_nodes: list

class HumanReviewEvent(Event):
    query: str
    reranked_nodes: list

class SynthesizeEvent(Event):
    query: str
    approved_nodes: list

class FeedbackEvent(Event):
    query: str
    answer: str

class ErrorEvent(StopEvent):
    error_message: str

# --- Workflow Definition ---
class AdvancedRAGWorkflow(Workflow):

    @step
    async def ingest(self, ctx: Context, ev: IngestionEvent) -> RetrievalEvent:
        logger.info("Starting ingestion...")
        documents = []
        # Text ingestion and chunking
        for path in ev.filepaths:
            try:
                with open(path, "r") as f:
                    text = f.read()
                chunks = [
                    text[i:i+ev.chunk_size]
                    for i in range(0, len(text), ev.chunk_size - ev.overlap)
                ]
                for idx, chunk in enumerate(chunks):
                    doc = Document(
                        text=chunk,
                        metadata={"source": path, "chunk_id": idx, "modality": "text"}
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to ingest {path}: {e}")
                return ErrorEvent(error_message=f"Ingestion failed: {e}")

        # Multi-modal: Image ingestion (store as metadata, could extend with CLIP or BLIP for embeddings)
        if ev.imagepaths:
            for imgpath in ev.imagepaths:
                try:
                    img = Image.open(imgpath)
                    doc = Document(
                        text=f"[IMAGE] {imgpath}",
                        metadata={"source": imgpath, "modality": "image"}
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to ingest image {imgpath}: {e}")
                    return ErrorEvent(error_message=f"Image ingestion failed: {e}")

        index = VectorStoreIndex.from_documents(documents)
        logger.info(f"Ingested {len(documents)} chunks from {len(ev.filepaths)} files and {len(ev.imagepaths or [])} images.")
        await ctx.store.set("index", index)
        return RetrievalEvent(query="", index=index)

    @step
    async def retrieve(self, ctx: Context, ev: RetrievalEvent) -> RerankEvent:
        logger.info(f"Retrieving with mode: {ev.retrieval_mode}")
        index = ev.index
        nodes = []
        try:
            if ev.retrieval_mode == "vector":
                retriever = index.as_retriever(similarity_top_k=10)
                nodes = retriever.retrieve(ev.query)
            elif ev.retrieval_mode == "keyword":
                nodes = [node for node in index.docstore.docs.values() if ev.query.lower() in node.text.lower()]
                nodes = nodes[:10]
            elif ev.retrieval_mode == "hybrid":
                retriever = index.as_retriever(similarity_top_k=7)
                vector_nodes = retriever.retrieve(ev.query)
                keyword_nodes = [node for node in index.docstore.docs.values() if ev.query.lower() in node.text.lower()]
                all_nodes = {id(node): node for node in vector_nodes + keyword_nodes}
                nodes = list(all_nodes.values())[:10]
            else:
                raise ValueError("Unknown retrieval mode")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return ErrorEvent(error_message=f"Retrieval failed: {e}")

        logger.info(f"Retrieved {len(nodes)} nodes.")
        await ctx.store.set("retrieved_nodes", nodes)
        return RerankEvent(query=ev.query, retrieved_nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RerankEvent) -> HumanReviewEvent:
        logger.info("Reranking...")
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=3,
            llm=OpenAI(model="gpt-4o-mini")
        )
        try:
            reranked_nodes = reranker.postprocess_nodes(ev.retrieved_nodes, query_str=ev.query)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return ErrorEvent(error_message=f"Reranking failed: {e}")

        logger.info(f"Reranked to top {len(reranked_nodes)} nodes.")
        await ctx.store.set("reranked_nodes", reranked_nodes)
        return HumanReviewEvent(query=ev.query, reranked_nodes=reranked_nodes)

    @step
    async def human_review(self, ctx: Context, ev: HumanReviewEvent) -> SynthesizeEvent:
        logger.info("Human-in-the-loop review step. Awaiting approval...")
        # Simulate human review: in production, replace this with UI or feedback API
        # For demo, auto-approve all nodes, but you could filter or edit
        approved_nodes = ev.reranked_nodes  # Replace with human-approved subset if needed
        logger.info(f"Human approved {len(approved_nodes)} nodes.")
        return SynthesizeEvent(query=ev.query, approved_nodes=approved_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: SynthesizeEvent) -> FeedbackEvent:
        logger.info("Synthesizing via external API...")
        context_texts = [node.get_content() for node in ev.approved_nodes]
        payload = {
            "query": ev.query,
            "contexts": context_texts
        }
        api_url = "https://api.external-llm.com/v1/synthesize"  # Replace with your endpoint

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload) as resp:
                    if resp.status != 200:
                        msg = f"External API error: {resp.status}"
                        logger.error(msg)
                        return ErrorEvent(error_message=msg)
                    data = await resp.json()
                    synthesized_answer = data.get("answer", "No answer returned by API.")
        except Exception as e:
            logger.error(f"External synthesis failed: {e}")
            return ErrorEvent(error_message=f"External synthesis failed: {e}")

        logger.info("Synthesis complete.")
        return FeedbackEvent(query=ev.query, answer=synthesized_answer)

    @step
    async def feedback(self, ctx: Context, ev: FeedbackEvent) -> StopEvent:
        logger.info("Collecting user feedback...")
        # Simulate feedback: in production, collect from UI or API
        print("SYNTHESIZED ANSWER:")
        print(ev.answer)
        feedback = input("Was this answer helpful? (yes/no): ").strip().lower()
        logger.info(f"User feedback: {feedback}")
        return StopEvent(result={"answer": ev.answer, "feedback": feedback})

# --- Entrypoint ---
async def main():
    # Example file paths (replace with your actual files)
    filepaths = ["data/doc1.txt", "data/doc2.txt"]
    imagepaths = ["data/img1.png"]  # Optional, for multi-modal

    workflow = AdvancedRAGWorkflow(timeout=120, verbose=True)
    ingestion_event = IngestionEvent(filepaths=filepaths, imagepaths=imagepaths)

    # Use checkpointing for observability
    checkpointer = WorkflowCheckpointer(workflow=workflow)
    result_handler = checkpointer.run(event=ingestion_event)
    result = await result_handler

    # Visualization: draw possible flows and most recent execution
    draw_all_possible_flows(AdvancedRAGWorkflow, filename="advanced_rag_workflow_all.html")
    draw_most_recent_execution(workflow, filename="advanced_rag_workflow_recent.html")

    if isinstance(result, ErrorEvent):
        print(f"Workflow failed: {result.error_message}")
    else:
        print("Final synthesized answer and feedback:")
        print(result.result)

if __name__ == "__main__":
    asyncio.run(main())
