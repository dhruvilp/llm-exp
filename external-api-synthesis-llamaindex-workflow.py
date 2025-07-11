import asyncio
import requests  # For HTTP API calls
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# --- Custom Event Definitions ---

class RetrieveEvent(StartEvent):
    query: str
    index: VectorStoreIndex

class SynthesizeEvent(StartEvent):
    query: str
    retrieved_nodes: list

# --- Custom Synthesis Step with External API Call ---

@step
async def external_synthesize(self, ev: SynthesizeEvent) -> StopEvent:
    # Prepare the context to send to your external API
    context_texts = [node.get_content() for node in ev.retrieved_nodes]
    payload = {
        "query": ev.query,
        "contexts": context_texts
    }
    # Replace this URL with your actual external API endpoint
    api_url = "https://api.external-llm.com/v1/synthesize"
    # Example: POST request to the external API (use your own authentication as needed)
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    synthesized_answer = response.json().get("answer", "No answer returned by API.")
    return StopEvent(result=synthesized_answer)

# --- Workflow Definition ---

class CustomRAGWorkflow(Workflow):
    @step
    async def retrieve(self, ev: RetrieveEvent) -> SynthesizeEvent:
        retriever = ev.index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(ev.query)
        return SynthesizeEvent(query=ev.query, retrieved_nodes=nodes)

    external_synthesize = external_synthesize  # Attach the custom synthesis step

# --- Entrypoint ---

async def main():
    # Ingest and index your documents
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Initialize workflow
    workflow = CustomRAGWorkflow()

    # Run the workflow with a user query
    query = "What are the health benefits of regular exercise?"
    result = await workflow.run(event=RetrieveEvent(query=query, index=index))
    print("Synthesized answer from external API:")
    print(result.result)

if __name__ == "__main__":
    asyncio.run(main())
