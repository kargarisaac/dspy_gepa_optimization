# Multi-Agent Medical Q&A System

A sophisticated medical question-answering system built with DSPy that uses specialized AI agents for diabetes and COPD queries, with document retrieval from PDF sources.

## Architecture

The system implements a hierarchical multi-agent architecture:

- **Diabetes Agent**: Specialized ReAct agent trained on diabetes literature
- **COPD Agent**: Specialized ReAct agent trained on COPD literature  
- **Lead Agent**: Meta-agent that routes questions to appropriate domain experts

## Key Features

- **PDF Document Processing**: Extracts and chunks medical PDFs using LangChain
- **Vector Search**: FAISS-based semantic search with HuggingFace embeddings
- **Agent Optimization**: Uses GEPA (DSPy's optimization technique) to improve agent performance
- **Multi-level Evaluation**: Custom LLM-based metrics for both step-level and answer-level assessment
- **Hierarchical Routing**: Lead agent intelligently delegates questions to domain experts

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
export OPENROUTER_API_KEY="your_key_here"
```

3. Place PDF documents in the `docs/` folder:
- `diabets1.pdf`, `diabets2.pdf` (diabetes literature)
- `copd1.pdf`, `copd2.pdf` (COPD literature)

4. Prepare Q&A datasets:
- `docs/qa_pairs_diabets.json`
- `docs/qa_pairs_copd.json`
- `docs/qa_pairs_joint.json`

## Usage

Run the script to:
1. Build FAISS vector databases from PDFs
2. Train and optimize domain-specific agents
3. Create and optimize the lead routing agent
4. Save optimized models to `dspy_program/`

The system automatically evaluates baseline vs optimized performance using medical Q&A datasets.

## File Structure

```bash
multi-agent-system.py    # Main implementation
docs/                    # PDF documents and Q&A datasets
faiss_index/            # Generated vector databases
    diabetes/
    copd/
dspy_program/           # Optimized agent models
    optimized_react_diabets.json
    optimized_react_copd.json
    optimized_lead_react.json
```

## How It Works

1. **Document Processing**: PDFs are loaded, chunked, and embedded into FAISS vector stores
2. **Agent Training**: Individual agents are created for each domain and optimized using GEPA
3. **Lead Agent**: A meta-agent learns to route questions to the appropriate domain expert
4. **Evaluation**: Performance is measured using LLM-based judges for medical accuracy

The system uses OpenRouter's GPT models for both the agents and evaluation metrics, ensuring high-quality medical responses.