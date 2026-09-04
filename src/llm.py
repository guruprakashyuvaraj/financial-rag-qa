"""
LLM factory. Wraps OpenAI chat models and AWS Bedrock models behind a
single get_llm() so the RAG chain doesn't care which provider is active.
"""
from src.config import settings


def get_llm(temperature: float = 0.1):
    if settings.llm_provider == "ollama":
        # Free, local LLM. Requires Ollama running (https://ollama.com) with
        # the model already pulled, e.g. `ollama pull llama3.2:1b`.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    if settings.llm_provider == "bedrock":
        from langchain_community.chat_models import BedrockChat
        import boto3

        client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        return BedrockChat(
            client=client,
            model_id=settings.bedrock_model_id,
            model_kwargs={"temperature": temperature},
        )

    # Default: OpenAI
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=temperature,
    )
