"""Multi-provider LLM factory for creating language models."""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from loguru import logger

from app.core.settings import settings
from app.loaders.base import AgentInfo

ModelType = Literal["gpt", "claude", "ollama", "gemini", "grok", "friendli", "luxia"]


def get_model_type_from_model_name(model: str) -> ModelType:
    """Infer model type from model name.

    Args:
        model: Model name string

    Returns:
        ModelType
    """
    model_lower = model.lower()

    if model_lower.startswith(("gpt-", "o1", "chatgpt")):
        return "gpt"
    elif model_lower.startswith(("claude-",)):
        return "claude"
    elif model_lower.startswith(("gemini-",)):
        return "gemini"
    elif model_lower.startswith(("grok-",)):
        return "grok"
    elif model_lower.startswith(("llama", "mistral", "codellama", "phi")):
        return "ollama"
    elif model_lower.startswith(("friendli-",)):
        return "friendli"
    elif model_lower.startswith(("luxia-",)):
        return "luxia"

    return settings.default_model_type


def create_openai_model(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool = True,
    **kwargs: Any,
) -> BaseChatModel:
    """Create OpenAI ChatGPT model.

    Args:
        model: Model name (e.g., gpt-4o, gpt-4-turbo)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        streaming: Enable streaming
        **kwargs: Additional model parameters

    Returns:
        ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        streaming=streaming,
        **kwargs,
    )


def create_anthropic_model(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool = True,
    **kwargs: Any,
) -> BaseChatModel:
    """Create Anthropic Claude model.

    Args:
        model: Model name (e.g., claude-3-5-sonnet-20241022)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        streaming: Enable streaming
        **kwargs: Additional model parameters

    Returns:
        ChatAnthropic instance
    """
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key,
        streaming=streaming,
        **kwargs,
    )


def create_ollama_model(
    model: str,
    temperature: float,
    **kwargs: Any,
) -> BaseChatModel:
    """Create Ollama model for local LLMs.

    Args:
        model: Model name (e.g., llama3.2, mistral)
        temperature: Sampling temperature
        **kwargs: Additional model parameters

    Returns:
        ChatOllama instance
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
        **kwargs,
    )


def create_gemini_model(
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> BaseChatModel:
    """Create Google Gemini model.

    Args:
        model: Model name (e.g., gemini-1.5-pro)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional model parameters

    Returns:
        ChatGoogleGenerativeAI instance
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=settings.google_api_key,
        **kwargs,
    )


def create_grok_model(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool = True,
    **kwargs: Any,
) -> BaseChatModel:
    """Create xAI Grok model.

    Args:
        model: Model name (e.g., grok-2-latest)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        streaming: Enable streaming
        **kwargs: Additional model parameters

    Returns:
        ChatXAI instance
    """
    from langchain_xai import ChatXAI

    return ChatXAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.xai_api_key,
        streaming=streaming,
        **kwargs,
    )


def create_friendli_model(
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> BaseChatModel:
    """Create Friendli model.

    Args:
        model: Model endpoint
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional model parameters

    Returns:
        ChatOpenAI instance configured for Friendli
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.friendli_api_key,
        base_url="https://api.friendli.ai/v1",
        **kwargs,
    )


def create_luxia_model(
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> BaseChatModel:
    """Create Luxia model.

    Args:
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional model parameters

    Returns:
        ChatOpenAI instance configured for Luxia
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.luxia_api_key,
        base_url=settings.luxia_base_url,
        **kwargs,
    )


def create_model_from_agent_info(
    agent_info: AgentInfo,
    streaming: bool = True,
) -> BaseChatModel:
    """Create LLM instance from agent configuration.

    Args:
        agent_info: Agent configuration containing model settings
        streaming: Enable streaming (if supported)

    Returns:
        BaseChatModel instance

    Raises:
        ValueError: If model type is not supported
    """
    model = agent_info.model
    temperature = agent_info.temperature
    max_tokens = agent_info.max_tokens

    # Get model type from metadata or infer from model name
    model_type = agent_info.metadata.get("model_type")
    if not model_type:
        model_type = get_model_type_from_model_name(model)

    logger.debug(f"Creating {model_type} model: {model}")

    if model_type == "gpt":
        return create_openai_model(model, temperature, max_tokens, streaming)
    elif model_type == "claude":
        return create_anthropic_model(model, temperature, max_tokens, streaming)
    elif model_type == "ollama":
        return create_ollama_model(model, temperature)
    elif model_type == "gemini":
        return create_gemini_model(model, temperature, max_tokens)
    elif model_type == "grok":
        return create_grok_model(model, temperature, max_tokens, streaming)
    elif model_type == "friendli":
        return create_friendli_model(model, temperature, max_tokens)
    elif model_type == "luxia":
        return create_luxia_model(model, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_model(
    model: str,
    model_type: ModelType | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    streaming: bool = True,
    **kwargs: Any,
) -> BaseChatModel:
    """Create LLM instance directly.

    Args:
        model: Model name
        model_type: Model type (if None, inferred from model name)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        streaming: Enable streaming
        **kwargs: Additional model parameters

    Returns:
        BaseChatModel instance
    """
    if model_type is None:
        model_type = get_model_type_from_model_name(model)

    logger.debug(f"Creating {model_type} model: {model}")

    if model_type == "gpt":
        return create_openai_model(model, temperature, max_tokens, streaming, **kwargs)
    elif model_type == "claude":
        return create_anthropic_model(model, temperature, max_tokens, streaming, **kwargs)
    elif model_type == "ollama":
        return create_ollama_model(model, temperature, **kwargs)
    elif model_type == "gemini":
        return create_gemini_model(model, temperature, max_tokens, **kwargs)
    elif model_type == "grok":
        return create_grok_model(model, temperature, max_tokens, streaming, **kwargs)
    elif model_type == "friendli":
        return create_friendli_model(model, temperature, max_tokens, **kwargs)
    elif model_type == "luxia":
        return create_luxia_model(model, temperature, max_tokens, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
