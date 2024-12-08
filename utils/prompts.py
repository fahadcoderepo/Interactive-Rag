"""
Custom LangChain prompt templates for RAG agent
"""

from langchain.prompts import PromptTemplate

def generate_qa_prompt_template() -> PromptTemplate:
    """Creates a prompt template for retrieval QA chain."""
    
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
    Context:
    {context}

    Question: {question}

    Answer:"""

    return PromptTemplate(template=template, input_variables=["context", "question"])

def generate_agent_prompt_templates() -> tuple:
    """Creates the agent's prompts with proper input variables."""
    
    # Define the system message and context for the agent
    prefix = """You are a helpful AI assistant with access to tools for answering questions.

Available tools:
{tools}

Your goal is to help users by:
1. Using tools to find relevant information
2. Providing clear and accurate answers
3. Asking for clarification when needed"""

    # Define the format instructions for tool usage
    format_instructions = """When using tools, follow this format:

Thought: what you're thinking about how to solve this
Action: the tool to use, must be one of [{tool_names}]
Action Input: the exact input to the tool
Observation: the tool's response
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I now know the final answer
Final Answer: the complete answer to the user's question"""

    # Define how to handle the conversation flow
    suffix = """Previous conversation:
{chat_history}

New question: {input}
{agent_scratchpad}"""

    return prefix, format_instructions, suffix