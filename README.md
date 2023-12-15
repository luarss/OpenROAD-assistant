# OpenROAD-assistant

Objective is to build a LLM Chatbot that is able to answer user queries for OpenROAD.

## Quickstart

```
# select chunking strategy
python chunk.py --chunk-size 200 --overlap 20

# regenerate embeddings (if loading old embedding)
python chat.py --gen-embeddings --max-tokens 200 --temp 0.2 --n-ctx 8000
```

## TODOs
- Test code-focused LLM vs instruct-focused LLM
- Gather RAG dataset
  - github issues
  - slack qna
- Oobabooga integration?
