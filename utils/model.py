"""
Utilities for model parameters configuration for Claude 3 Sonnet
"""
from typing import Dict, Any, Optional

def get_model_params(
    model_id: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Set up model parameters for Claude 3 Sonnet

    Parameters
    ----------
    model_id : str
        Model identifier (anthropic.claude-3-sonnet-20240229-v1:0)
    params : dict
        Base parameters containing:
        - answer_length: maximum tokens in response
        - temperature: randomness (0-1)
        - top_p: nucleus sampling parameter (0-1)
        - stop_sequences: list of strings to stop generation

    Returns
    -------
    Dict[str, Any]
        Model-specific parameters for Claude 3

    Raises
    ------
    ValueError
        If unsupported model ID is provided
    """
    if model_id != "anthropic.claude-3-sonnet-20240229-v1:0":
        raise ValueError(
            f"Unsupported model: {model_id}. "
            "Only anthropic.claude-3-sonnet-20240229-v1:0 is currently supported."
        )
    
    # Get parameters with defaults
    max_tokens = params.get("answer_length", 4096)
    temperature = params.get("temperature", 0.0)
    top_p = params.get("top_p", 0.9)
    stop_sequences = params.get("stop_sequences", ["\n\nHuman:"])

    # Validate parameters
    if not 0 <= temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1")
    if not 0 <= top_p <= 1:
        raise ValueError("Top_p must be between 0 and 1")
    if max_tokens < 1:
        raise ValueError("Max tokens must be positive")

    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop_sequences": stop_sequences,
        "messages": []  # Required for Claude chat format
    }