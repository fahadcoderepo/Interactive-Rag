"""
Streamlit frontend for RAG Agent Demo with Claude 3 Sonnet
"""

import boto3
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.chat_models import BedrockChat
from langchain_community.retrievers import AmazonKendraRetriever

from utils.userInput import UserInteractionTool
from utils.model import get_model_params
from utils.prompts import generate_qa_prompt_template, generate_agent_prompt_templates

# Page Configuration
st.set_page_config(
    page_title="RAG Agent Demo",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown("### Enhancing LLM Responses with Clarification Tools")

# Clear session state if needed
if 'user_answer' in st.session_state:
    del st.session_state['user_answer']

# Agent type selection
mode = st.selectbox(
    label="Select agent type",
    options=("Agent with UserInteractionTool", "Agent wiht Traditional RAG Agent"),
)

### PARAMETERS
MODEL_REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
PARAMS = {
    "answer_length": 4096,
    "temperature": 0.0,
    "top_p": 0.9,
    "stop_sequences": ["\n\nHuman:"]
}

# Retriever params
KENDRA_INDEX_ID = st.secrets.get("KENDRA_INDEX_ID")
TOP_K = 3
MIN_CONFIDENCE = 0.1

# Memory params
MEMORY_SIZE = 3

@st.cache_resource
def initialize_components():
    """Initialize and cache the main components"""
    try:
        # Initialize Kendra retriever with proper attribute filter
        retriever = AmazonKendraRetriever(
            index_id=KENDRA_INDEX_ID,
            top_k=TOP_K,
            region_name=MODEL_REGION,
            attribute_filter={
                "EqualsTo": {
                    "Key": "_language_code",
                    "Value": {
                        "StringValue": "en"
                    }
                }
            },
            min_score_confidence=MIN_CONFIDENCE
        )
        
        # Initialize Bedrock client
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=MODEL_REGION,
            endpoint_url=f'https://bedrock-runtime.{MODEL_REGION}.amazonaws.com'
        )
        
        # Initialize LLM
        model_params = get_model_params(model_id=MODEL_ID, params=PARAMS)
        llm = BedrockChat(
            client=bedrock_client,
            model_id=MODEL_ID,
            model_kwargs=model_params,
            streaming=True
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": generate_qa_prompt_template(),
            },
        )
        
        return llm, qa_chain, retriever
    except Exception as e:
        st.error(f"Error in initialization: {str(e)}")
        raise

# Initialize components
try:
    llm, qa_chain, retriever = initialize_components()
except Exception as e:
    st.error(f"Error initializing components: {str(e)}")
    st.stop()

def qa_chain_wrapper(question: str) -> str:
    """Wrapper for QA chain to handle multiple outputs"""
    try:
        result = qa_chain({"query": question})
        return result['result']
    except Exception as e:
        st.error(f"Error in retrieval: {str(e)}")
        return f"Error in retrieval: {str(e)}"

# Set up conversation memory
conversational_memory = ConversationBufferMemory(
    memory_key="chat_history",
    k=MEMORY_SIZE,
    return_messages=True,
)

# Create tools
kendra_tool = Tool(
    name="KendraRetrievalTool",
    func=qa_chain_wrapper,
    description="Use this tool to find answers from the knowledge base. Input should be a question.",
)

human_ask_tool = UserInteractionTool()

# Create tools list based on mode
tools = [human_ask_tool, kendra_tool] if mode == "Agent with AskHuman tool" else [kendra_tool]

# Get prompt templates
prefix, format_instructions, suffix = generate_agent_prompt_templates()

# Format tools description for the prompt
tools_description = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)

# Initialize agent
try:
    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=5,
        early_stopping_method="generate",
        memory=conversational_memory,
        agent_kwargs={
            "prefix": prefix.format(tools=tools_description),
            "format_instructions": format_instructions,
            "suffix": suffix,
        },
    )
except Exception as e:
    st.error(f"Error initializing agent: {str(e)}")
    st.stop()

# Initialize chat history if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("Ask your question"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(prompt, callbacks=[st_callback])
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Failed to get agent response: {str(e)}")
            if 'user_answer' in st.session_state:
                del st.session_state['user_answer']