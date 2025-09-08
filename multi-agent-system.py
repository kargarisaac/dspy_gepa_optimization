# %% [markdown]
# # Workshop Overview: Language Models, Transformers, and Practical RAG + GEPA

# This workshop mixes big-picture concepts with hands-on code. We'll start with short,
# beginner-friendly explanations, then build a Retrieval-Augmented Generation (RAG)
# system and optimize an agent using DSPy's GEPA optimizer.

# References mentioned throughout:
# - GEPA hands-on reranker notebook (structure/inspiration):
#   [GEPA-Hands-On-Reranker](https://github.com/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/dspy/GEPA-Hands-On-Reranker.ipynb)
# - DSPy documentation: `https://dspy.ai/`
# - GEPA paper overview: [alphaXiv: GEPA](https://www.alphaxiv.org/overview/2507.19457v1)

# ## Introduction – What are Language Models?

# Language Models (LMs) are probabilistic programs trained to predict the next token
# given previous tokens. With enough data and compute, they learn useful world
# knowledge and procedures (e.g., answering questions, following instructions). You
# interact with them by sending prompts; they generate text continuations.

# Core ideas:
# - Tokenization: text → tokens → vectors
# - Next-token prediction objective during pretraining
# - Emergent capabilities from scale (reasoning, translation, Q&A)


# ### Quick code sample: Querying an LM (after configuration below)
# %%

import dspy
import os
from dotenv import load_dotenv
# from dspy.adapters.baml_adapter import BAMLAdapter
import json
from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from pathlib import Path
import random

# LangChain document loaders, text splitter, vectorstore and embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings   # wrapper for sentence-transformers
from langchain.vectorstores import FAISS
from langchain.schema import Document

# import mlflow

# mlflow.dspy.autolog(
#     log_compiles=True,    # Track optimization process
#     log_evals=True,       # Track evaluation results
#     log_traces_from_compile=True  # Track program traces during optimization
# )

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("med-ai-workshop")
# mlflow.autolog()

load_dotenv()

# %%
question = "What is a language model in one sentence?"
lm = dspy.LM(
    "openai/gpt-4.1-mini",                 # LiteLLM route for OpenRouter models
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model_type="chat",
    cache=False,
    temperature=0.3,
    max_tokens=20000
)

print(lm(question))


# %%
# After we configure `lm` later in this notebook:
question = "What is a language model in one sentence?"
lm = dspy.LM(
    "openrouter/openai/gpt-oss-20b",                 # LiteLLM route for OpenRouter models
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model_type="chat",
    cache=False,
    temperature=0.3,
    max_tokens=20000
)

print(lm(question))

# %%
# local_lm = dspy.LM(
#         model='ollama_chat/qwen3:4b',
#         api_base='http://localhost:11434',
#         api_key=''
#     )

# print(local_lm(question))

# %% [markdown]
# ## Transformer Basics (simple)

# Transformers are the dominant architecture for LLMs. At a high level:
# - Embeddings turn tokens into vectors
# - Self-attention lets each token attend to others to collect relevant context
# - Position information (e.g., positional encodings/rotaries) preserves order
# - Stacked layers produce contextualized representations; the final head predicts next tokens

# For inference, we feed a prompt and generate tokens step-by-step, optionally using
# techniques like temperature and top-p for diversity.


# ## Training and Fine-Tuning Large Language Models

# - Pretraining: LMs learn general language/statistical patterns from large corpora
# - Supervised fine-tuning (SFT): train on instruction-following or task-specific data
# - Preference optimization (e.g., RLHF/DPO): align outputs with human preferences
# - Programmatic prompting: optimize prompts/programs without changing model weights
#   (this is what we do here with GEPA: evolve the program text with a teacher LM)

# Note: Parameter-efficient finetuning (e.g., LoRA) adjusts a small set of weights.
# GEPA, by contrast, evolves instructions/demos (text), not model parameters.

# ### Pretraining (self-supervised foundation)

# Pretraining teaches an LM general language competence via next-token prediction
# (causal LM) on large corpora. Objective: maximize log-likelihood of observed text.

# Key ingredients:
# - Massive, diverse text; careful filtering/deduplication
# - Causal objective (next-token) with curriculum, stable optimizers, scaling laws
# - Tokenizer (BPE, sentencepiece); long-context training if budget allows

# Minimalized objective (conceptual):
# ```python
# # Given tokens x[0..T]
# loss = 0
# for t in range(1, T+1):
#     loss += - log p_theta(x[t] | x[:t])
# optimize(theta, loss)
# ```


# ### Instruction Following (SFT)

# Supervised Fine-Tuning (SFT) aligns a pretrained model to follow instructions.
# Data consists of (instruction, input, output) triples, often with system messages.

# Tips:
# - Use clean, diverse instructions; ensure consistent formatting/schema
# - Include few-shot exemplars for structured outputs and tool schemas
# - Validate with held-out instructions; monitor length and failure modes

# Example (conceptual JSONL):
# ```json
# {"system": "You are helpful.",
#  "instruction": "Summarize in one sentence",
#  "input": "LLMs predict next tokens.",
#  "output": "LLMs generate text by predicting the next token."}
# ```


# ### Chain-of-Thought (CoT) training

# CoT teaches step-by-step reasoning by supervising intermediate rationales before
# the final answer. This can be done via SFT (rationales in outputs) and/or process
# rewards (PRM) in RL.

# Approaches:
# - CoT SFT: outputs contain a reasoning field + final answer; caution with privacy
# - Short CoT / self-consistency: sample multiple rationales, pick majority answer
# - PRM: reward good steps without exposing full thoughts at inference

# DSPy example signature for CoT:
# ```python
# class Solve(dspy.Signature):
#     problem = dspy.InputField()
#     reasoning = dspy.OutputField()
#     answer = dspy.OutputField()

# solver = dspy.ChainOfThought(Solve)
# res = solver(problem="If 12*8=96, what is 96+5?")
# print(res.reasoning, res.answer)
# ```


# ### Reinforcement Learning and Reasoning for LLMs (o3, DeepSeek, Kimi K2)

# Beyond SFT, modern reasoning models use RL to improve multi-step thinking,
# tool-use, and robustness. While implementation details vary by vendor, the
# common recipe is:

# 1) Pretrain → SFT: Teach the model the basic interface (instructions, tool schemas,
#    examples). For tool use, SFT includes step-by-step traces with function calls.
# 2) RL for reasoning: Optimize for task success in an environment (unit tests,
#    verifiable tasks, tool outcomes), guided by rewards from outcome and/or process
#    judges (PRMs). This reduces superficial pattern-matching and improves planning.

# Notes on specific families (high level):
# - OpenAI o3: Focuses on improved reasoning via post-training with RL and outcome/
#   process feedback on verifiable tasks. Emphasizes correctness under a budget and
#   safer, more consistent behavior.
# - DeepSeek (e.g., R1/R1-Zero): Uses RL to strengthen long-form reasoning. Combines
#   outcome supervision (tests) and process reward models to reward useful steps
#   along the chain-of-thought without always exposing full reasoning at inference.
# - Moonshot Kimi K2: Post-trained for robust tool-use and reasoning. Trained to call
#   tools (including MCP-style tool servers) via SFT on tool traces and RL in a live
#   environment, rewarding valid calls, correct arguments, successful results, and
#   accurate final answers.

# Reward design (typical):
# - Outcome rewards: pass/fail against tests, exact match, numerical tolerance, BLEU.
# - Process rewards (PRM): judge intermediate steps (reasoning quality, valid tool
#   invocations, schema adherence). Encourages good planning, not just lucky outputs.
# - Cost/latency penalties: discourage excessive calls or long traces.

# Training for tool calling
# - SFT: Provide supervised trajectories that include tool JSON calls and results.
#   The model learns when to emit a tool_call vs a final answer.
# - RL: Place the model in an environment with real tools (search, calculator, DB).
#   Give positive reward for successful calls and correct final answers; penalize
#   invalid calls, schema errors, or irrelevant tools.

# Minimalized pseudo-formats (illustrative only):

# ```json
# // SFT example of a tool-using dialog
# {
#   "system": "You may call tools by emitting tool_call JSON.",
#   "tools": [
#     {"name": "search", "schema": {"q": "string"}},
#     {"name": "calculator", "schema": {"expr": "string"}}
#   ],
#   "messages": [
#     {"role": "user", "content": "What is (12*8)+5?"},
#     {"role": "assistant", "tool_call": {"name": "calculator", "arguments": {"expr": "12*8"}}},
#     {"role": "tool", "name": "calculator", "result": {"value": 96}},
#     {"role": "assistant", "final_answer": "101"}
#   ]
# }
# ```

# ```python
# # RL loop sketch (policy gradient-style)
# env = ToolEnvironment(tools=[calculator, search])
# policy = initialize_policy_from_sft()
# for epoch in range(E):
#     traj = env.rollout(policy, prompt)
#     reward = evaluate(traj)  # outcome + process + cost/latency
#     policy = update(policy, traj, reward)  # e.g., PPO/GRPO-style step
# ```

# Takeaways
# - SFT gets you a usable baseline for tools and instruction following.
# - RL refines when/why to call tools, improves planning, and aligns to verifiable
#   objectives (pass tests, exact match, success@k). This is how models like o3,
#   DeepSeek R1, and Kimi K2 make reasoning/tool-use more reliable in practice.



# ## Key Applications of LLMs

# - Chat assistants and copilots
# - Retrieval-Augmented Generation (RAG) for grounded answers
# - Agents with tools (search, code execution, retrieval) and planning (e.g., ReAct)
# - Summarization, extraction, reranking, classification, and more

# In this workshop, we build a RAG pipeline with FAISS + LangChain and an agentic
# ReAct solver that calls a retrieval tool.
 


# ## What this workshop covers in code (mapping to topics)

# - RAG with local vector DB: we build a FAISS index using LangChain and
#   Sentence Transformers, then retrieve passages for questions.
# - Agentic reasoning (ReAct): the agent uses a retrieval tool (`vector_search_tool`).
# - Programmatic optimization with GEPA: we evolve the agent's program text using a
#   teacher LM for reflection, and a separate metric LM for judging.
# - Evaluation: we use `dspy.Evaluate` and a metric that returns `dspy.Prediction`.

# Topics we explain conceptually but do not change in model weights:
# - Transformer internals and parametric finetuning are summarized, but we focus on
#   prompt/program evolution (GEPA) rather than training the model parameters.


# # Vector DB Creation

# In this workshop, we'll build a small Retrieval-Augmented Generation (RAG) system
# step by step and then optimize an agent with DSPy's GEPA optimizer. We start by:
# - Loading two local PDF papers with LangChain
# - Splitting pages into overlapping chunks for dense retrieval
# - Creating embeddings with a Sentence Transformers model
# - Storing vectors in a local FAISS index for fast similarity search

# Why these choices?
# - FAISS is a fast, proven vector database for approximate nearest neighbor search.
# - Overlapping chunks preserve context across boundaries for better retrieval.
# - Sentence Transformers (MiniLM) provide lightweight, strong embeddings suitable for CPU.

# References for this section:
# - LangChain vector stores and document loaders: https://python.langchain.com
# - FAISS library: https://faiss.ai

# %%
"""
Build a FAISS vector DB from two local PDF papers using LangChain.
Outputs a directory "faiss_index" with the persisted vectorstore.
"""

# --- Config: change these paths to your downloaded PDFs ---
DIABETES_PDF_PATHS = ["docs/diabets1.pdf", "docs/diabets2.pdf"]   # <-- put your two PDF filenames here
COPD_PDF_PATHS = ["docs/copd1.pdf", "docs/copd2.pdf"]
OUTPUT_DIABETES_FAISS_DIR = "faiss_index/diabetes"
OUTPUT_COPD_FAISS_DIR = "faiss_index/copd"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# chunk settings (tweak for your needs)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 200

def load_pdfs(paths):
    """Load PDFs into LangChain Document objects (keeps page-level granularity)."""
    all_docs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        loader = PyPDFLoader(str(p))
        # load returns a list of Document objects (one per page typically)
        pages = loader.load()
        # add a source filename into metadata for traceability
        for i, doc in enumerate(pages):
            # ensure a copy of metadata dict (avoid mutating shared objects)
            meta = dict(doc.metadata or {})
            meta["source"] = str(p.name)
            meta["page"] = i
            doc.metadata = meta
        all_docs.extend(pages)
    return all_docs

# %% [markdown]
# ## Loading PDFs (beginner-friendly overview)

# - We use `PyPDFLoader` to read each PDF into page-level `Document` objects.
# - We add `source` and `page` metadata to every `Document` to keep track of provenance.
# - Keeping page granularity helps trace answers back to the original PDF.

def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks (keeps metadata)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    # split_documents returns list[Document] (with page_content and metadata)
    chunks = text_splitter.split_documents(documents)
    return chunks

# %% [markdown]
# ## Why chunking matters

# LLMs answer better when given concise, focused passages. We split long pages into
# overlapping chunks so that:
# - Important sentences near boundaries are not lost (`chunk_overlap`)
# - Retrieval returns smaller, more relevant contexts for the model.

def build_vectorstore(chunks, model_name=EMBEDDING_MODEL, save_dir=OUTPUT_DIABETES_FAISS_DIR):
    """Create embeddings and store them in a FAISS vectorstore, then persist to disk."""
    # Instantiate HuggingFaceEmbeddings wrapper (requires sentence-transformers installed)
    hf_emb = HuggingFaceEmbeddings(model_name=model_name,
                                   model_kwargs={"device": "cpu"})  # change to "cuda" if available

    # Build FAISS index from LangChain Document objects
    print("Creating FAISS vector store from", len(chunks), "chunks. This may take a while...")
    vectorstore = FAISS.from_documents(chunks, hf_emb)

    # Persist to disk
    vectorstore.save_local(save_dir)
    print(f"Saved FAISS vectorstore to: {save_dir}")
    return vectorstore, hf_emb

# %% [markdown]
# ## From text to vectors

# - `HuggingFaceEmbeddings` wraps Sentence Transformers to compute dense vectors.
# - `FAISS.from_documents` builds the index directly from `Document` chunks.
# - We persist the FAISS index locally so it can be reused across runs.


# %%

# if db exists, load it
print("Loading Diabetes PDFs...")
docs = load_pdfs(DIABETES_PDF_PATHS)
print(f"Loaded {len(docs)} page-documents from {len(DIABETES_PDF_PATHS)} PDFs.")

print("Chunking Diabetes documents...")
chunks = chunk_documents(docs)
print(f"Produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

diabetes_vectorstore, diabetes_embeddings = build_vectorstore(chunks, save_dir=OUTPUT_DIABETES_FAISS_DIR)

# %%

print("Loading COPD PDFs...")
docs = load_pdfs(COPD_PDF_PATHS)
print(f"Loaded {len(docs)} page-documents from {len(COPD_PDF_PATHS)} PDFs.")

print("Chunking COPD documents...")
chunks = chunk_documents(docs)
print(f"Produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

copd_vectorstore, copd_embeddings = build_vectorstore(chunks, save_dir=OUTPUT_COPD_FAISS_DIR)


# %%
def diabetes_vector_search_tool(query: str, k: int = 3) -> str:
    """
    A tool for the ReAct agent.
    Performs vector search and returns a formatted string of results.
    """
    results = diabetes_vectorstore.similarity_search_with_score(query, k=k)
    context = ""
    for i, (doc, score) in enumerate(results):
        doc_content = doc.page_content
        context += f"[PASSAGE {i+1}, score={score:.4f}]\n{doc_content}\\n\\n"
    return context

# %% 

def copd_vector_search_tool(query: str, k: int = 3) -> str:
    """
    A tool for the ReAct agent.
    Performs vector search and returns a formatted string of results.
    """
    results = copd_vectorstore.similarity_search_with_score(query, k=k)
    context = ""
    for i, (doc, score) in enumerate(results):
        doc_content = doc.page_content
        context += f"[PASSAGE {i+1}, score={score:.4f}]\n{doc_content}\\n\\n"
    return context


# %% [markdown]
# ## Retrieval helper
#
# `vector_search` performs a similarity search for a query and returns the top-k
# passages with scores. We'll use this as a tool for our agent later.

# quick retrieval test
diabetes_vector_search_tool("What are the main treatments for Type 2 diabetes?", k=3)

# %%
copd_vector_search_tool("What are the main treatments for COPD?", k=3)


# %% [markdown]
# # Simple RAG
#
# In RAG, we retrieve relevant passages first, then ask the LLM to answer using them.
# We'll set up two LMs:
# - A "student" LM (`lm`) used to answer questions
# - A "teacher" LM (`teacher_lm`) used later by GEPA to reflect and propose improvements
#
# We will use OpenRouter here; set `OPENROUTER_API_KEY` as an environment variable.
# You can switch to other providers by changing `api_base` and `model`.

# %%


# Configure your LM (DSPy tutorial uses dspy.LM)
lm = dspy.LM(
    "openrouter/openai/gpt-oss-20b",                 # LiteLLM route for OpenRouter models
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model_type="chat",
    cache=False,
    temperature=0.3,
    max_tokens=64000
)
dspy.settings.configure(lm=lm)

# Teacher LM for reflection (GEPA)
teacher_lm = dspy.LM(
    "openrouter/openai/gpt-oss-120b",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model_type="chat",
    cache=False,
    temperature=0.3,
    max_tokens=64000
)

# %% [markdown]
# ## The RAG signature (inputs/outputs)
#
# We define a small DSPy signature `RAGQA` with:
# - `question` and retrieved `context` as inputs
# - `answer` as the output
#
# DSPy composes programs from signatures; `ChainOfThought` adds reasoning steps.

# %%
# Define a signature (simple QA)
class RAGQA(dspy.Signature):
    """You are a helpful assistant. Answer a question using retrieved passages"""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


# %%
rag = dspy.ChainOfThought(RAGQA)

# %%
question = "What is Gestational Diabetes Mellitus (GDM)?"
retrieved_context = diabetes_vector_search_tool(question, k=3)


# %%
rag(context=retrieved_context, question=question)

# %%
lm.inspect_history(n=1)


# %% [markdown]
# # Agentic RAG
#
# ReAct (Reason + Act) lets an LLM think step-by-step and use tools. Here, the tool
# is our `vector_search` retriever. The agent decides when to call the tool, then
# uses the returned passages to produce an answer.

# %%
react = dspy.ReAct(signature="question->answer", tools=[diabetes_vector_search_tool])
question = "What is Gestational Diabetes Mellitus (GDM)?"
pred = react(question=question)

# %% 

pred

# %%
lm.inspect_history(n=1)


# %% [markdown]
# # Evaluation and Optimization of ReAct Agent
#
# In this section, we will:
# 1. Load a small Q&A dataset from `docs/qa_pair.json`.
# 2. Evaluate the baseline ReAct agent using a metric that returns `dspy.Prediction`.
# 3. Optimize the agent with DSPy's GEPA optimizer using a teacher LM for reflection.
# 4. Re-evaluate the optimized agent on the same dev set.
#
# References to follow along:
# - GEPA hands-on reranker (structure/inspiration):
#   [GEPA notebook](https://github.com/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/dspy/GEPA-Hands-On-Reranker.ipynb)
# - DSPy documentation (metrics, Evaluate, teleprompters): [DSPy docs](https://dspy.ai/)

# %%
# Load the dataset
with open("docs/qa_pairs_diabets.json", "r") as f:
    qa_diabetes_data = json.load(f)

# Convert to dspy.Example objects
dataset_diabetes = [dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question") for item in qa_diabetes_data]

# shuffle the dataset
random.shuffle(dataset_diabetes)

# Split the dataset as requested
train_size = 20
trainset_diabetes = dataset_diabetes[:train_size]
devset_diabetes = dataset_diabetes[train_size:]

print(f"Loaded {len(dataset_diabetes)} examples.")
print(f"Train set size: {len(trainset_diabetes)}")
print(f"Dev set size: {len(devset_diabetes)}")

# %%
# Load the dataset
with open("docs/qa_pairs_copd.json", "r") as f:
    qa_copd_data = json.load(f)

# Convert to dspy.Example objects
dataset_copd = [dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question") for item in qa_copd_data]

# shuffle the dataset
random.shuffle(dataset_copd)

# Split the dataset as requested
trainset_copd = dataset_copd[:10]
devset_copd = dataset_copd[10:]

print(f"Loaded {len(dataset_copd)} examples.")
print(f"Train set size: {len(trainset_copd)}")
print(f"Dev set size: {len(devset_copd)}")

# %% [markdown]
# ## Why we wrap the retriever as a tool
#
# Agents do better when tools return clean, readable text. We'll wrap `vector_search`
# into a `vector_search_tool` that formats passages nicely for the LLM.



# %% [markdown]
# ## Baseline agent for evaluation
#
# We'll keep two agents:
# - `react`: the earlier ReAct agent that uses `vector_search`
# - `unoptimized_react`: identical behavior but uses the formatted `vector_search_tool`
#
# We'll evaluate the baseline first to get a reference point.

# %%
# Define the metric for evaluation using an LLM to check for factual consistency.
class JudgeConsistency(dspy.Signature):
    """Judge whether the predicted answer matches the gold answer.

    # Instructions:
    - The score should be between 0.0 and 1.0 and based on the similarity of the predicted answer and the gold answer.
    - The justification should be a brief explanation of the score.
    - If the answer doesn't address the question properly, the score should be less than 0.5.
    - If the answer is completely correct, the score should be 1.0. Otherwise, the score should be less than 1.0.
    - Be very strict in your judgement as this is a medical question.
    """
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="a float score between 0 and 1")
    justification: str = dspy.OutputField()

def llm_metric_prediction(*args, **kwargs):
    """Metric returning ScoreWithFeedback for GEPA and Evaluate.

    Accepts flexible arguments because GEPA may pass additional positional
    parameters (e.g., program, trace, batch metadata). We only need `example`
    and `pred` here; the rest are ignored.
    """
    # Extract example and prediction from positional/keyword args
    example = kwargs.get("example") or kwargs.get("gold")
    pred = kwargs.get("pred") or kwargs.get("prediction")
    if example is None and len(args) > 0:
        example = args[0]
    if pred is None and len(args) > 1:
        pred = args[1]

    # Defensive checks
    if example is None or pred is None:
        return ScoreWithFeedback(score=0.0, feedback="Missing example or pred")

    predicted_answer = getattr(pred, "answer", None) or ""
    if not predicted_answer.strip():
        return ScoreWithFeedback(score=0.0, feedback="Empty prediction")

    # Deterministic judge to ensure module and feedback scores match
    with dspy.settings.context(lm=lm):
        judge = dspy.Predict(JudgeConsistency)
        judged = judge(
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=predicted_answer,
        )

    score = getattr(judged, "score", None) or 0.0
    # clip the score to be between 0 and 1
    score = max(0.0, min(1.0, score))
    justification = getattr(judged, "justification", "") or ""
    feedback_text = f"Score: {score}. {justification}".strip()
    return ScoreWithFeedback(score=score, feedback=feedback_text)

# %% [markdown]
# ## Metric LM vs Reflection LM — when and why do we need both?
#
# - **Metric LM (Judge)**: used by `Evaluate`/GEPA to produce a structured verdict and
#   score for each candidate program. In our case, `llm_metric_prediction` returns a
#   `dspy.Prediction` with `score`, `verdict`, and `justification`. The metric LM should
#   be stable and strict, so we use the teacher LM here for judging.
# - **Reflection LM (Teacher)**: used by GEPA to critique failure cases and propose
#   edits (mutations) to the program's prompts/instructions/demos. It is not used to
#   score candidates; it generates suggestions that the metric will later evaluate.
#
# You can technically use a single LM for both roles, but separating them reduces
# evaluation bias and prevents the student from overfitting to its own evaluator.

# ## About the metric (important)
#
# - DSPy metrics can return a numeric score or a `dspy.Prediction` object.
# - GEPA works well with metrics that return `dspy.Prediction`, letting us carry
#   extra signals (like a verdict and justification).
# - Here we use a lightweight LLM judge (`JudgeConsistency`) to label predictions
#   as Yes/No relative to the gold answer. This is simple but surprisingly effective.
# - GEPA may call the metric with extra positional parameters (e.g., `program`,
#   `trace`, or batch metadata). Our metric accepts `*args, **kwargs` and extracts
#   only `(example, pred)` so it works seamlessly with both `Evaluate` and GEPA.


# %%
# Set up the evaluator
evaluator_diabetes = Evaluate(devset=devset_diabetes, num_threads=32, display_progress=True, display_table=5, provide_traceback=True)

# %%
class ReActSignature(dspy.Signature):
    """You are a helpful assistant. Answer user's question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class DiabetesAgent(dspy.Module):
    def __init__(
        self
    ):
        super().__init__()
        # init LLM
        self.lm = dspy.LM(
            "openrouter/openai/gpt-oss-20b", 
            api_key=os.getenv("OPENROUTER_API_KEY"), 
            temperature=0.3, 
            max_tokens=64000,
            cache=False
        )
        dspy.configure(lm=self.lm)
        self.agent = dspy.ReAct(ReActSignature, tools=[diabetes_vector_search_tool])

    def forward(self, question: str):
        return self.agent(question=question)

diabetes_agent = DiabetesAgent()


# %%
diabetes_agent(question="What are the main treatments for Type 2 diabetes?")

# %%
diabetes_agent.lm.inspect_history(n=2)

# %% 
diabetes_agent

# %%
# Evaluate the baseline agent (the existing `react`)
print("Evaluating the baseline ReAct agent...")
with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    diabetes_baseline_eval = evaluator_diabetes(diabetes_agent, metric=llm_metric_prediction)

# %%
diabetes_baseline_eval

# %% [markdown]
# # Optimizing the ReAct Agent with GEPA
#
# GEPA (Generative Evolutionary Prompt Adaptation) automatically improves programs
# by evolving their textual components (e.g., instructions, demonstrations) using
# LLM-based reflection. Key knobs used below:
# - `auto`: budget preset (light/medium/heavy) controlling search breadth and depth
# - `reflection_lm`: the teacher LM that critiques and proposes improvements
# - `reflection_minibatch_size`: how many examples to bundle per reflection round
# - `num_threads`: parallelism for faster evaluation and reflection
# - `track_stats`: collect basic stats about the search for later inspection
#
# Reference notebook for GEPA concepts and flow:
# [GEPA hands-on reranker](https://github.com/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/dspy/GEPA-Hands-On-Reranker.ipynb)

# ## GEPA pseudo-code (high-level)
#
# The following is a simplified, conceptual outline of the GEPA workflow. It shows
# the separation of roles between the metric (judge) and the reflection LM, and how
# the candidate pool evolves under a fixed budget.
#
# ```python
# # Seed: initial program (student) that uses our retrieval tool
# student = dspy.ReAct("question -> answer", tools=[vector_search_tool])
#
# # Metric (judge) returns dspy.Prediction; we run it under teacher LM for stability
# def metric(example, pred):
#     with dspy.settings.context(lm=teacher_lm):
#         return llm_metric_prediction(example, pred)
#
# # Initialize GEPA with metric and reflection LM (teacher)
# teleprompter = GEPA(
#     metric=metric,
#     reflection_lm=teacher_lm,
#     auto="medium",
#     num_threads=16,
# )
#
# # Initialize candidate pool (population) from the seed student
# population = [student]  # plus internal variations/mutations created by GEPA
# budget = teleprompter.budget()  # derived from `auto` preset and internal knobs
#
# while budget.remaining():
#     # Evaluate each candidate on a (mini)batch from train/dev
#     scores = {prog: Evaluate(devset=batch, metric=metric)(prog)
#               for prog, batch in rollout(population)}
#
#     # Find failure cases and ask the reflection LM (teacher) to critique
#     # and propose concrete prompt/program edits
#     reflections = reflect_with_teacher(population, scores, teacher_lm)
#
#     # Apply edits to produce new candidates, then select survivors on Pareto front
#     population = select_and_mutate(population, scores, reflections)
#
# # Pick the best candidate after the budget is exhausted
# optimized = best_candidate(population, scores)
# ```

# ## GEPA in practice: from a seed prompt to an evolved program
#
# 1. **Seed prompt/program**: Start with your initial DSPy program (here, a `ReAct`
#    agent with the `vector_search_tool`). Conceptually, this contains a seed
#    instruction like:
#
#    "You are a ReAct agent with access to a retrieval tool. Decide when to call
#    `vector_search_tool(question)` to gather evidence. Then answer concisely using
#    retrieved passages. Prefer precise, factual statements."
#
# 2. **Population initialization**: GEPA creates an initial pool of candidate
#    variants (mutations) of your program's textual components (instructions,
#    demos, few-shot exemplars).
#
# 3. **Rollout + scoring (budget)**: Each candidate is evaluated on a subset of
#    training/dev examples under a global budget (e.g., the `auto` preset controls
#    the number of metric/reflection calls). The metric LM produces `dspy.Prediction`
#    scores to rank candidates.
#
# 4. **Reflection (teacher LM)**: For poor-performing cases, the reflection LM reads
#    failures (often in minibatches) and writes critiques + concrete edits to the
#    candidate's prompts/demos. This yields improved candidates.
#
# 5. **Selection and survival**: Candidates are selected using multi-objective criteria
#    (e.g., performance, compactness). GEPA keeps the best (Pareto-front) candidates
#    and discards the rest, updating the pool.
#
# 6. **Iteration**: Steps 3–5 repeat until the budget is exhausted or convergence.
#    The final best candidate becomes your optimized program.
#
# Practical tips:
# - Keep the metric LM separate from the student LM.
# - Use a stronger/stricter teacher for reflection when possible.
# - Tune the `auto` preset or budget-related flags based on your compute limits.

# %%
def react_metric(*args, **kwargs):
    pred_name = kwargs.get("pred_name")
    pred_trace = kwargs.get("pred_trace")
    gold = kwargs.get("gold")
    pred = kwargs.get("pred")
    # pred_name is 'react' or 'extract.predict'
    if pred_name == "react":
        # Example heuristic: penalize early 'finish', missing JSON, or not using the vector tool
        step = pred_trace[0][2]  # Prediction dict for this predictor
        tool = step.get("next_tool_name", "")
        args = step.get("next_tool_args", {})
        ok_json = isinstance(args, dict) and ("query" in args)
        used_search = tool == "diabetes_vector_search_tool"
        finished = tool == "finish"

        score = 0.0
        if used_search and ok_json:
            score += 0.6
        if not finished:
            score += 0.2
        # Write actionable feedback that rewrites react instructions
        fb = []
        if not used_search:
            fb.append("Always call diabetes_vector_search_tool at least once before finish.")
        if not ok_json:
            fb.append("Ensure next_tool_args is valid JSON with a 'query' key.")
        if finished:
            fb.append("Avoid selecting 'finish' until you have ≥3 relevant passages.")
        if not fb:
            fb.append("Good. Keep queries concise and set k=5 by default.")

        return dspy.Prediction(score=score, feedback=" ".join(fb))

    # Fallback: overall answer quality (e.g., exact match / F1)
    return llm_metric_prediction(*args, **kwargs)


# %%
diabetes_agent.agent.extract._compiled = False
diabetes_agent.agent.react._compiled = False
# Set up the teleprompter/optimizer using GEPA (per reference notebook)
teleprompter = GEPA(
    metric=react_metric, #llm_metric_prediction,
    # max_metric_calls=2,
    max_full_evals=2,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    # reflection_minibatch_size=5,
    add_format_failure_as_feedback=True,
    reflection_lm=teacher_lm,
    # use_merge=True,
    # max_merge_invocations=5,
)

# %%
# Compile/Optimize the ReAct agent (use the friendlier tool for better LM usability)
with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    optimized_diabetes_agent = teleprompter.compile(student=diabetes_agent, trainset=trainset_diabetes, valset=devset_diabetes)

# %%
optimized_diabetes_agent


# %%
# Access the detailed results from your optimized agent
results = optimized_diabetes_agent.detailed_results

# %% 
# Get all candidates and their validation scores
candidates = results.candidates
candidates 

# %%
val_scores = results.val_aggregate_scores
val_scores 

# %% 
# Find the best candidate by validation score
best_idx = results.best_idx  # This is automatically calculated
best_score = val_scores[best_idx]
best_candidate = results.best_candidate

print(f"Best candidate index: {best_idx}")
print(f"Best validation score: {best_score}")
print(f"Best candidate components: {best_candidate}")

# %%
# List all candidates with their scores (sorted by performance)
print("\nAll candidates ranked by validation score:")
for rank, (idx, score) in enumerate(sorted(enumerate(val_scores), key=lambda x: x[1], reverse=True), 1):
    print(f"Rank {rank}: Candidate {idx} - Score: {score}")

# %%


# %%
print("\\n\\nEvaluating the optimized ReAct agent...")
with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    optimized_diabetes_eval = evaluator_diabetes(optimized_diabetes_agent, metric=llm_metric_prediction)

# %%
optimized_diabetes_eval

# %%
optimized_diabetes_agent(question="What are the main treatments for Type 2 diabetes?")

# %%
optimized_diabetes_agent.lm.inspect_history(n=2)

# %%
optimized_diabetes_agent

# %%
# save the optimized model in a new folder
os.makedirs("dspy_program", exist_ok=True)
optimized_diabetes_agent.save("dspy_program/optimized_react_diabets.json", save_program=False)

# %% [markdown]
# ## What GEPA does under the hood (intuitively)
#
# 1. Start with your program (student agent)
# 2. Propose variations (mutations) to the program's text instructions/demos
# 3. Use the teacher LM to reflect on failures and guide better mutations
# 4. Evaluate candidates on the train/dev sets with our metric
# 5. Keep and refine the best candidates (evolutionary search)

# # Evaluating the Optimized Agent

# ## Compare baseline vs optimized
#
# After optimization, we evaluate on the same dev set to see if the score improves.
# You can also inspect the model's call history to understand its reasoning.


# # Multi-Agent Extension: COPD Specialist, Lead Orchestrator, and Joint Optimization
#
# In this section, we add a COPD RAG agent, wrap both domain agents as callable tools
# for a lead ReAct agent, and then optimize the lead agent on a joint dataset that
# concatenates Diabetes and COPD questions and answers.

# %%
# Instantiate the COPD expert agent
class COPDAgent(dspy.Module):
    def __init__(self):
        self.lm = dspy.LM(
            "openrouter/openai/gpt-oss-20b", 
            api_key=os.getenv("OPENROUTER_API_KEY"), 
            temperature=0.3, 
            max_tokens=64000,
            cache=False
        )
        dspy.configure(lm=self.lm)
        self.copd_agent = dspy.ReAct(ReActSignature, tools=[copd_vector_search_tool])

    def forward(self, question: str):
        return self.copd_agent(question=question)
    
copd_agent = COPDAgent()


# %%
evaluator_copd = Evaluate(devset=devset_copd, num_threads=32, display_progress=True, display_table=5, provide_traceback=True)

# Evaluate the baseline agent (the existing `react`)
print("Evaluating the baseline ReAct agent...")
with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    copd_baseline_eval = evaluator_copd(copd_agent, metric=llm_metric_prediction)

# %%
copd_baseline_eval

# %%
# %%
teleprompter = GEPA(
    metric=llm_metric_prediction,
    # max_metric_calls=100,
    max_full_evals=1,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    # reflection_minibatch_size=5,
    reflection_lm=teacher_lm,
    # use_merge=True,
    # max_merge_invocations=5,
)

with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    optimized_copd_agent = teleprompter.compile(student=copd_agent, trainset=trainset_copd, valset=devset_copd)

# %%
print("\\n\\nEvaluating the optimized ReAct agent...")
with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    optimized_copd_eval = evaluator_copd(optimized_copd_agent, metric=llm_metric_prediction)

# %% 
optimized_copd_eval

# %%
# save the optimized model in a new folder
os.makedirs("dspy_program", exist_ok=True)
optimized_copd_agent.save("dspy_program/optimized_react_copd.json", save_program=False)


# %%

# Wrap the domain agents as callable tools for the lead agent
# Prefer the optimized Diabetes agent if available; otherwise fallback to baseline `react`.

def ask_diabetes(question: str) -> str:
    """Call the Diabetes expert agent and return its answer text."""
    pred = optimized_diabetes_agent(question=question)
    return pred.answer


def ask_copd(question: str) -> str:
    """Call the COPD expert agent and return its answer text."""
    pred = optimized_copd_agent(question=question)
    return pred.answer


# Lead ReAct agent that can call sub-agents as tools
class LeadReAct(dspy.Module):
    def __init__(self):
        self.lm = dspy.LM(
            "openrouter/openai/gpt-oss-20b", 
            api_key=os.getenv("OPENROUTER_API_KEY"), 
            temperature=0.3, 
            max_tokens=64000,
            cache=False
        )
        dspy.configure(lm=self.lm)
        self.lead_react = dspy.ReAct(ReActSignature, tools=[ask_diabetes, ask_copd])

    def forward(self, question: str):
        return self.lead_react(question=question)
    
lead_react = LeadReAct()

# %%
# Load the dataset
with open("docs/qa_pairs_joint.json", "r") as f:
    qa_joint_data = json.load(f)

# Convert to dspy.Example objects
joint_dataset = [dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question") for item in qa_joint_data]

# shuffle the dataset
random.shuffle(joint_dataset)

# Split the dataset as requested
trainset_joint = joint_dataset[:train_size]
devset_joint = joint_dataset[train_size:]

print(f"Loaded {len(joint_dataset)} examples.")
print(f"Train set size: {len(trainset_joint)}")
print(f"Dev set size: {len(devset_joint)}")


# %%
# Baseline evaluation of the lead agent on the joint dev set
evaluator_joint = Evaluate(devset=devset_joint, num_threads=32, display_progress=True, display_table=5, provide_traceback=True)
print("Evaluating baseline Lead ReAct (agents-as-tools) on joint dev set...")
with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    baseline_lead_eval = evaluator_joint(lead_react, metric=llm_metric_prediction)

# %%
baseline_lead_eval

# %%

teleprompter_joint = GEPA(
    metric=llm_metric_prediction,
    # max_metric_calls=100,
    max_full_evals=1,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    # reflection_minibatch_size=5,
    reflection_lm=teacher_lm,
    # use_merge=True,
    # max_merge_invocations=5,
)

with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    optimized_lead_react = teleprompter_joint.compile(student=lead_react, trainset=trainset_joint, valset=devset_joint)

# %%
print("\n\nEvaluating the optimized Lead ReAct agent...")
with dspy.settings.context(bypass_suggest=True, bypass_assert=True):
    optimized_lead_eval = evaluator_joint(optimized_lead_react, metric=llm_metric_prediction)

# %%
optimized_lead_eval

# %%
lead_react(question="What are the main treatments for Type 2 diabetes?")
lead_react.lm.inspect_history(n=2)


# %%
optimized_lead_react(question="What are the main treatments for Type 2 diabetes?")
optimized_lead_react.lm.inspect_history(n=2)


# %%
# Save the optimized lead agent
optimized_lead_react.save("dspy_program/optimized_lead_react.json", save_program=False)


# %% [markdown]
# ## Challenges and Limitations (recap)
#
# - Hallucinations without grounding; rely on retrieval and calibrated judges
# - Limited context windows; chunking and retrieval parameters matter
# - Evaluation bias; prefer separating student vs judge/reflection LMs
# - Cost/latency; consider batching, caching, smaller models for iteration
# - Privacy/safety; filter tools/outputs, handle PII carefully

# %% [markdown]
# ## Future Directions and Closing Notes
#
# - Better evaluators (multi-judge, calibrated feedback, test-time self-checks)
# - Multi-agent systems and tool ecosystems
# - Long-context models and memory architectures
# - Hybrid reasoning (symbolic + neural)
# - Programmatic optimization (GEPA) to continually evolve prompts/programs

# %% 
