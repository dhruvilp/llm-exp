from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

###
query = "What are the health benefits of regular exercise?"
retriever = index.as_retriever(similarity_top_k=10)
retrieved_nodes = retriever.retrieve(query)

###
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor.llm_rerank import LLMRerank

reranker = LLMRerank(
    choice_batch_size=5,  # Number of candidates per LLM call
    top_n=3,              # Number of top results to keep
    llm=OpenAI(model="gpt-4o-mini")  # Specify your LLM
)
reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=query)

###
from llama_index.core.response_synthesizers import CompactAndRefine

llm = OpenAI(model="gpt-4o-mini")
synthesizer = CompactAndRefine(llm=llm, streaming=True)
response = synthesizer.synthesize(query, nodes=reranked_nodes)
print(response)

###
from llama_index.core.workflow import Workflow, step, Context, StartEvent, StopEvent

class RAGWorkflow(Workflow):
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent):
        # ... retrieve logic ...
        return RetrieverEvent(nodes=nodes)
    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent):
        # ... rerank logic using LLMRerank ...
        return RerankEvent(nodes=new_nodes)
    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent):
        # ... synthesis logic ...
        return StopEvent(result=response)





