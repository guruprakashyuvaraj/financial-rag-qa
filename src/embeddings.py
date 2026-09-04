"""
Embedding model factory. Supports OpenAI and AWS Bedrock embedding models
behind a single get_embeddings() call so the rest of the app is provider-agnostic.
"""
from langchain_openai import OpenAIEmbeddings

from src.config import settings


def get_embeddings():
    if settings.embedding_provider == "huggingface":
        # Free, local, no API key required. Downloads the model once (~80MB)
        # and runs entirely on CPU.
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)

    if settings.embedding_provider == "bedrock":
        from langchain_community.embeddings import BedrockEmbeddings
        import boto3

        client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        return BedrockEmbeddings(
            client=client,
            model_id=settings.bedrock_embedding_model_id,
        )

    # Default: OpenAI
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
