import time
from typing import Any, Optional

from langchain.tools.base import BaseTool
from pydantic import Field
import streamlit as st


class UserInteractionTool(BaseTool):
    """
    A custom LangChain tool for human-driven interaction,
    enabling query clarification when KendraRetrievalTool fails to find an answer.
    """

    name: str = Field(default="UserInteractionTool")
    description: str = Field(
        default="Prompts user for clarification or missing details when no answer is found."
    )

    def _run(self, user_query: str, run_manager: Optional[Any] = None) -> str:
        """
        Handles the interaction by prompting the user for input.
        Waits until the user provides a response.
        Args:
            user_query (str): The query to clarify.
            run_manager (Optional[Any]): Context manager, not currently utilized.
        
        Returns:
            str: User's input response.
        """
        
        # Check if session state is initialized for storing user responses
        if "user_response" not in st.session_state:
            # Display query to user in a chat-like assistant interface
            assistant_message = st.chat_message("assistant", avatar="🧑")
            assistant_message.write(user_query)

            # Wait for user input
            user_input = st.text_input("Please provide your response:", key="response_input")
            while not user_input:  # Keep checking until user provides a valid input
                time.sleep(0.5)

            # Store user input into session state for retrieval later
            st.session_state["user_response"] = user_input

        return st.session_state["user_response"]

    def _arun(self, user_query: str) -> str:
        """
        Async execution is not currently supported for this tool.
        Args:
            user_query (str): Query string to process asynchronously.
        
        Raises:
            NotImplementedError: Indicates async is unsupported.
        """
        raise NotImplementedError("Asynchronous execution is not supported by this tool.")
