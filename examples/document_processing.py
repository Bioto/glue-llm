"""Document processing pipeline example using GlueLLM workflows.

This example demonstrates:
- Pipeline workflow for document processing
- Multi-stage processing (extract, analyze, summarize)
- Tool integration for file operations
- Error handling in workflows
"""

import asyncio

from gluellm.executors import AgentExecutor
from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt
from gluellm.workflows.pipeline import PipelineWorkflow


# Simulated file operations (in production, use real file I/O)
class DocumentStore:
    """Simulated document storage."""

    def __init__(self):
        self.documents = {
            "doc1.txt": "Python is a high-level programming language. It's known for simplicity and readability.",
            "doc2.txt": "Machine learning is a subset of artificial intelligence that enables systems to learn.",
            "doc3.txt": "Cloud computing provides on-demand computing resources over the internet.",
        }

    def read_document(self, filename: str) -> str:
        """Read a document from storage."""
        return self.documents.get(filename, f"Document {filename} not found.")

    def save_extracted_data(self, filename: str, data: str) -> str:
        """Save extracted data."""
        return f"Saved extracted data from {filename} to database."


doc_store = DocumentStore()


# Tool functions
def read_document(filename: str) -> str:
    """Read a document file.

    Args:
        filename: Name of the document file

    Returns:
        Document content
    """
    return doc_store.read_document(filename)


def save_extracted_data(filename: str, data: str) -> str:
    """Save extracted data to database.

    Args:
        filename: Source document filename
        data: Extracted data to save

    Returns:
        Confirmation message
    """
    return doc_store.save_extracted_data(filename, data)


async def document_processing_example():
    """Run document processing pipeline example."""
    print("=" * 60)
    print("Document Processing Pipeline Example")
    print("=" * 60)

    # Stage 1: Document Extractor
    extractor = Agent(
        name="Document Extractor",
        description="Extracts and structures content from documents",
        system_prompt=SystemPrompt(
            content="""You are a document extraction specialist.
            Read documents and extract key information in a structured format.
            Focus on main topics, key facts, and important details."""
        ),
        tools=[read_document],
        max_tool_iterations=3,
    )

    # Stage 2: Content Analyzer
    analyzer = Agent(
        name="Content Analyzer",
        description="Analyzes extracted content for insights",
        system_prompt=SystemPrompt(
            content="""You are a content analyst.
            Analyze extracted content and identify:
            - Main themes and topics
            - Key insights and patterns
            - Important relationships between concepts
            Provide a comprehensive analysis."""
        ),
        tools=[],
        max_tool_iterations=3,
    )

    # Stage 3: Summarizer
    summarizer = Agent(
        name="Summarizer",
        description="Creates concise summaries from analysis",
        system_prompt=SystemPrompt(
            content="""You are a summarization expert.
            Create clear, concise summaries that capture:
            - Main points
            - Key insights
            - Actionable takeaways
            Keep summaries under 200 words."""
        ),
        tools=[save_extracted_data],
        max_tool_iterations=3,
    )

    # Create pipeline workflow
    pipeline = PipelineWorkflow(
        stages=[
            ("extract", AgentExecutor(extractor)),
            ("analyze", AgentExecutor(analyzer)),
            ("summarize", AgentExecutor(summarizer)),
        ]
    )

    # Process documents
    documents = ["doc1.txt", "doc2.txt", "doc3.txt"]

    for doc_name in documents:
        print(f"\n{'=' * 60}")
        print(f"Processing: {doc_name}")
        print("=" * 60)

        input_prompt = f"Process the document: {doc_name}"

        try:
            result = await pipeline.execute(input_prompt)

            print(f"\nPipeline completed in {result.iterations} stages")
            print("\nFinal Summary:")
            print("-" * 60)
            print(result.final_output)
            print("-" * 60)

            # Show stage outputs
            print("\nStage Outputs:")
            for i, interaction in enumerate(result.agent_interactions, 1):
                stage = interaction.get("stage", "unknown")
                output_preview = interaction.get("output", "")[:100]
                print(f"  {i}. {stage}: {output_preview}...")

        except Exception as e:
            print(f"Error processing {doc_name}: {e}")
            # In production, log error and continue with next document

    print("\n" + "=" * 60)
    print("Document Processing Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(document_processing_example())
