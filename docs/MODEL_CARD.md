# Model Card: Legal AI

## Model Details

### Embedding Model
- **Name**: BAAI/bge-small-en-v1.5
- **Type**: Sentence Transformer (BERT-based)
- **Dimensions**: 384
- **Size**: ~100MB
- **License**: MIT
- **Source**: https://huggingface.co/BAAI/bge-small-en-v1.5
- **Usage**: Document and query embedding for vector search

### Reranker Model
- **Name**: BAAI/bge-reranker-base
- **Type**: Cross-encoder
- **Size**: ~1.1GB
- **License**: MIT
- **Source**: https://huggingface.co/BAAI/bge-reranker-base
- **Usage**: Re-ranking retrieval results for improved relevance

### Language Model
- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Quantization**: Q4_K_M (GGUF format)
- **Size**: ~4.7GB
- **License**: Llama 3.1 Community License
- **Source**: https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
- **Runtime**: llama-cpp-python (CPU, CUDA, or DirectML)
- **Context Window**: 4096 tokens

## Intended Use

- Legal document analysis and question answering
- Indian law research assistance
- Contract review and clause identification
- Legal document summarization

## Limitations

- The model is trained on general text corpora and fine-tuned on legal Q&A. It may not capture nuanced legal interpretations.
- Always verify citations against the original source documents.
- Not a substitute for professional legal advice.
- May exhibit biases present in the training data.
- Performance may vary on complex legal scenarios requiring specialized domain expertise.

## Bias and Fairness

The underlying LLM may reflect biases present in its training data. The legal documents indexed are from official government sources and may reflect systemic biases in the legal system.

## Evaluation

- Retrieval: Evaluated on a test set of 20 Indian legal Q&A pairs
- Generation: Evaluated for citation accuracy and factual grounding

## Training Data

- **Base model**: Llama 3.1 pre-training data (Meta)
- **Fine-tuning data**: Synthetic legal Q&A pairs generated from the indexed Indian legal corpora
- **Retrieval corpus**: Constitution of India, IPC, CrPC, Evidence Act, and other Central Acts
