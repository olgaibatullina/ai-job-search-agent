EMBEDDING_MODELS = {
    "mpnet": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-mpnet-base-v2",
    },
    "minilm": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "roberta": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-roberta-large-v1",
    },
    "bge": {
        "provider": "huggingface",
        "model": "BAAI/bge-large-en-v1.5",
    },
}