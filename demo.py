"""
Streamlit frontend for RAG Agent Demo with enhanced visibility
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

# Enhanced Page Configuration
st.set_page_config(
    page_title="Intelligent RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better visibility
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
        background-color: #ffffff;
    }
    .stApp {
        background-color: #ffffff;
    }
    
    /* Chat message containers */
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        font-size: 16px !important;
        color: #000000 !important;
        background-color: white;
    }
    
    /* User message styling */
    .user-message {
        background-color: #E3F2FD !important;
        border-left: 5px solid #1976D2;
        margin-left: 20px;
        margin-right: 60px;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #F5F5F5 !important;
        border-left: 5px solid #2E7D32;
        margin-left: 60px;
        margin-right: 20px;
    }
    
    /* Ensure text visibility */
    .stMarkdown, p, div[data-testid="stMarkdownContainer"] > p {
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1976D2;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        width: 100%;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1976D2;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("""
    <h1 style='text-align: center; margin-bottom: 2rem; color: #1976D2;'>
        🤖 Intelligent RAG Assistant
    </h1>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Configuration")
    st.divider()
    
    mode = st.selectbox(
        "Select Agent Type",
        ("Agent with AskHuman tool", "Traditional RAG Agent"),
        help="Choose how you want to interact with the assistant"
    )
    
    st.markdown("### Model Settings")
    MODEL_REGION = "us-east-1"
    MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9)
        max_tokens = st.slider("Max Tokens", 1000, 8000, 4096)

    def clear_chat():
        st.session_state.messages = []
        st.rerun()

    st.button("Clear Chat History", on_click=clear_chat)

# Parameters
PARAMS = {
    "answer_length": max_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "stop_sequences": ["\n\nHuman:"]
}

KENDRA_INDEX_ID = st.secrets.get("KENDRA_INDEX_ID")
TOP_K = 3
MIN_CONFIDENCE = 0.1
MEMORY_SIZE = 3

@st.cache_resource
def initialize_components():
    """Initialize and cache the main components"""
    try:
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
        
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=MODEL_REGION,
            endpoint_url=f'https://bedrock-runtime.{MODEL_REGION}.amazonaws.com'
        )
        
        model_params = get_model_params(model_id=MODEL_ID, params=PARAMS)
        llm = BedrockChat(
            client=bedrock_client,
            model_id=MODEL_ID,
            model_kwargs=model_params,
            streaming=True
        )
        
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
    """Wrapper for QA chain to handle responses"""
    try:
        result = qa_chain({"query": question})
        return result['result']
    except Exception as e:
        st.error(f"Error in retrieval: {str(e)}")
        return f"Error in retrieval: {str(e)}"

# Memory setup
conversational_memory = ConversationBufferMemory(
    memory_key="chat_history",
    k=MEMORY_SIZE,
    return_messages=True,
)

# Tools setup
kendra_tool = Tool(
    name="KendraRetrievalTool",
    func=qa_chain_wrapper,
    description="Use this tool to find answers from the knowledge base. Input should be a question.",
)

human_ask_tool = UserInteractionTool()
tools = [human_ask_tool, kendra_tool] if mode == "Agent with AskHuman tool" else [kendra_tool]

# Prompt setup
prefix, format_instructions, suffix = generate_agent_prompt_templates()
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

# Chat Interface
st.markdown("### Chat History")
chat_container = st.container()

# Initialize or get chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(
        message["role"], 
        avatar="👤" if message["role"] == "user" else "🤖"
    ):
        st.markdown(
            f'<div class="chat-message {"user-message" if message["role"] == "user" else "assistant-message"}">{message["content"]}</div>',
            unsafe_allow_html=True
        )

# Chat input and response handling
if prompt := st.chat_input("Ask your question..."):
    # Display user message
    st.chat_message("user", avatar="👤").markdown(
        f'<div class="chat-message user-message">{prompt}</div>',
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            with st.spinner('Processing your question...'):
                response = agent.run(prompt, callbacks=[st_callback])
                message_placeholder.markdown(
                    f'<div class="chat-message assistant-message">{response}</div>',
                    unsafe_allow_html=True
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Failed to get response: {str(e)}")
            if 'user_answer' in st.session_state:
                del st.session_state['user_answer']

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with Streamlit, LangChain, and Claude 3 🚀</p>
    </div>
""", unsafe_allow_html=True)