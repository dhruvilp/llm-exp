import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

# Set up the OpenAI client via Instructor
client = instructor.from_openai(OpenAI())

# Define the structured output models
class Label(BaseModel):
    chunk_id: int = Field(description="The unique identifier of the text chunk")
    chain_of_thought: str = Field(description="The reasoning process used to evaluate the relevance")
    relevancy: int = Field(
        description="Relevancy score from 0 to 10, where 10 is most relevant",
        ge=0,
        le=10,
    )

class RerankedResults(BaseModel):
    labels: list[Label] = Field(description="List of labeled and ranked chunks")

    @field_validator("labels")
    @classmethod
    def model_validate(cls, v: list[Label]) -> list[Label]:
        # Sort by relevancy descending
        return sorted(v, key=lambda x: x.relevancy, reverse=True)

# The reranker function
def rerank_results(query: str, chunks: list[dict]) -> RerankedResults:
    return client.chat.completions.create(
        model="gpt-4o-mini",  # You can use another model if you prefer
        response_model=RerankedResults,
        messages=[
            {
                "role": "system",
                "content": """
You are an expert search result ranker. Your task is to evaluate the relevance of each text chunk to the given query and assign a relevancy score.
For each chunk:
1. Analyze its content in relation to the query.
2. Provide a chain of thought explaining your reasoning.
3. Assign a relevancy score from 0 to 10, where 10 is most relevant.
Be objective and consistent in your evaluations.
"""
            },
            {
                "role": "user",
                "content": """
<query>{{ query }}</query>
<chunks_to_rank>
{% for chunk in chunks %}
<chunk id="{{ chunk.id }}">
{{ chunk.text }}
</chunk>
{% endfor %}
</chunks_to_rank>
Please provide a RerankedResults object with a Label for each chunk.
"""
            }
        ],
        context={"query": query, "chunks": chunks},
    )

# Example usage
def main():
    query = "What are the health benefits of regular exercise?"
    chunks = [
        {"id": 0, "text": "Regular exercise can improve cardiovascular health and reduce the risk of heart disease."},
        {"id": 1, "text": "The price of gym memberships varies widely depending on location and facilities."},
        {"id": 2, "text": "Exercise has been shown to boost mood and reduce symptoms of depression and anxiety."},
        {"id": 3, "text": "Proper nutrition is essential for maintaining a healthy lifestyle."},
        {"id": 4, "text": "Strength training can increase muscle mass and improve bone density, especially important as we age."},
    ]
    results = rerank_results(query, chunks)
    print("Reranked results:")
    for label in results.labels:
        print(f"Chunk {label.chunk_id} (Relevancy: {label.relevancy}):")
        print(f"Text: {chunks[label.chunk_id]['text']}")
        print(f"Reasoning: {label.chain_of_thought}\n")

if __name__ == "__main__":
    main()
