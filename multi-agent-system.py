# %%

import dspy
import os
from dotenv import load_dotenv
# from dspy.adapters.baml_adapter import BAMLAdapter
import json
import logging
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

# Configure basic logging if not already configured by the host app
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metric")

# %%
# question = "What is a language model in one sentence?"
# lm = dspy.LM(
#     "openai/gpt-4.1-mini",                 # LiteLLM route for OpenRouter models
#     api_base="https://api.openai.com/v1",
#     api_key=os.environ["OPENAI_API_KEY"],
#     model_type="chat",
#     cache=False,
#     temperature=0.3,
#     max_tokens=20000
# )

# print(lm(question))


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

# %%
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

# %%

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


# %%
# quick retrieval test
diabetes_vector_search_tool("What are the main treatments for Type 2 diabetes?", k=3)

# %%
copd_vector_search_tool("What are the main treatments for COPD?", k=3)

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
    "openrouter/openai/gpt-oss-20b",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model_type="chat",
    cache=False,
    temperature=0.3,
    max_tokens=64000
)

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

# %%
react = dspy.ReAct(signature="question->answer", tools=[diabetes_vector_search_tool])
question = "What is Gestational Diabetes Mellitus (GDM)?"
pred = react(question=question)

# %% 

pred

# %%
lm.inspect_history(n=1)

# %%
# Load the dataset
with open("docs/qa_pairs_diabets.json", "r") as f:
    qa_diabetes_data = json.load(f)

# Convert to dspy.Example objects
dataset_diabetes = [dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question") for item in qa_diabetes_data]

# shuffle the dataset
random.shuffle(dataset_diabetes)

# Split the dataset as requested
train_size = 8
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

class JudgeReactStep(dspy.Signature):
    """Judge whether the next tool call (name + args) is appropriate, well-formed, and relevant.

    - Output a strict score in [0, 1].
    - Provide a brief justification and a yes/no style verdict in justification text.
    """
    question: str = dspy.InputField()
    tool_name: str = dspy.InputField()
    tool_args_json: str = dspy.InputField()
    score: float = dspy.OutputField(desc="a float score between 0 and 1")
    verdict: str = dspy.OutputField()
    justification: str = dspy.OutputField()

def llm_metric_prediction(*args, **kwargs):
    """Metric returning ScoreWithFeedback for GEPA and Evaluate.

    Accepts flexible arguments because GEPA may pass additional positional
    parameters (e.g., program, trace, batch metadata). We only need `example`
    and `pred` here; the rest are ignored.
    """
    # GEPA may pass predictor context for per-predictor feedback
    pred_name = kwargs.get("pred_name")
    pred_trace = kwargs.get("pred_trace")
    if pred_name is not None:
        logger.info(f"metric called for predictor={pred_name}")

    # Extract example and prediction from positional/keyword args
    example = kwargs.get("example") or kwargs.get("gold")
    pred = kwargs.get("pred") or kwargs.get("prediction")
    if example is None and len(args) > 0:
        example = args[0]
    if pred is None and len(args) > 1:
        pred = args[1]

    # Special handling: when optimizing the ReAct loop predictor
    if pred_name and (pred_name == "react" or pred_name.endswith(".react")) and pred_trace:
        try:
            _, step_inputs, step_outputs = pred_trace[0]
        except Exception:
            step_inputs, step_outputs = {}, {}

        question_text = getattr(example, "question", None) or step_inputs.get("question", "") or ""

        # Read tool name/args from the predictor's outputs (dict or Prediction)
        def _get(o, key, default=""):
            if isinstance(o, dict):
                return o.get(key, default)
            return getattr(o, key, default)

        tool_name = _get(step_outputs, "next_tool_name", "")
        tool_args = _get(step_outputs, "next_tool_args", {})

        # Heuristics: well-formed JSON args and sensible fields
        args_is_dict = isinstance(tool_args, dict)
        has_query = args_is_dict and isinstance(tool_args.get("query"), str) and tool_args.get("query", "").strip() != ""
        k_val = tool_args.get("k") if args_is_dict else None
        k_ok = isinstance(k_val, int) and 1 <= k_val <= 10 or k_val is None
        used_tool = tool_name not in ("", "finish")
        early_finish = tool_name == "finish"

        logger.debug(
            "react-step details | used_tool=%s tool=%s args_keys=%s has_query=%s k=%s early_finish=%s pred_trace_len=%s",
            used_tool,
            str(tool_name),
            list(tool_args.keys()) if isinstance(tool_args, dict) else type(tool_args).__name__,
            has_query,
            k_val,
            early_finish,
            len(pred_trace) if pred_trace else 0,
        )

        heuristics_score = 0.0
        if used_tool:
            heuristics_score += 0.4
        if has_query:
            heuristics_score += 0.4
        if k_ok:
            heuristics_score += 0.1
        if not early_finish:
            heuristics_score += 0.1
        heuristics_score = max(0.0, min(1.0, heuristics_score))

        # LLM judge for the loop step (tool choice + query relevance)
        tool_args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
        with dspy.settings.context(lm=lm):
            react_judge = dspy.Predict(JudgeReactStep)
            judged = react_judge(
                question=question_text,
                tool_name=str(tool_name),
                tool_args_json=tool_args_json,
            )

        llm_score = getattr(judged, "score", 0.0) or 0.0
        llm_score = max(0.0, min(1.0, llm_score))
        llm_just = getattr(judged, "justification", "") or ""

        total = 0.5 * heuristics_score + 0.5 * llm_score

        logger.info(
            "react-step scores | heuristics=%.3f llm=%.3f total=%.3f",
            heuristics_score,
            llm_score,
            total,
        )

        # Actionable feedback
        suggestions = []
        if not used_tool:
            suggestions.append("Select a retrieval tool before finishing.")
        if early_finish:
            suggestions.append("Avoid selecting 'finish' until you have evidence from the retrieval tool.")
        if not args_is_dict:
            suggestions.append("Emit next_tool_args as a valid JSON object.")
        else:
            if not has_query:
                suggestions.append("Include a non-empty 'query' string in next_tool_args.")
            if k_val is not None and (not isinstance(k_val, int) or k_val < 1 or k_val > 10):
                suggestions.append("Choose a reasonable k (e.g., 3–5).")
        if not suggestions:
            suggestions.append("Good step. Keep queries concise and set k=5 by default.")

        feedback_text = (
            f"ReAct step — LLM score: {llm_score:.2f}, heuristics: {heuristics_score:.2f}. "
            + " ".join(suggestions)
            + (f" LLM justification: {llm_just}" if llm_just else "")
        ).strip()

        return ScoreWithFeedback(score=total, feedback=feedback_text)

    # Program-level or non-react predictor: judge final answer quality
    # Defensive checks
    if example is None or pred is None:
        return ScoreWithFeedback(score=0.0, feedback="Missing example or pred")

    predicted_answer = getattr(pred, "answer", None) or ""
    if not predicted_answer.strip():
        return ScoreWithFeedback(score=0.0, feedback="Empty prediction")

    with dspy.settings.context(lm=lm):
        judge = dspy.Predict(JudgeConsistency)
        judged = judge(
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=predicted_answer,
        )

    score = getattr(judged, "score", None) or 0.0
    score = max(0.0, min(1.0, score))
    justification = getattr(judged, "justification", "") or ""
    logger.info("answer-level score=%.3f for question='%s'", score, (example.question[:80] + "...") if len(example.question) > 80 else example.question)
    feedback_text = f"Score: {score}. {justification}".strip()
    return ScoreWithFeedback(score=score, feedback=feedback_text)

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
diabetes_baseline_eval = evaluator_diabetes(diabetes_agent, metric=llm_metric_prediction)

# %%
diabetes_baseline_eval


# %%
dspy.enable_logging()
diabetes_agent.agent.extract._compiled = True
diabetes_agent.agent.react._compiled = False
# Set up the teleprompter/optimizer using GEPA (per reference notebook)
teleprompter = GEPA(
    metric=llm_metric_prediction,
    max_full_evals=2,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    add_format_failure_as_feedback=True,
    reflection_lm=teacher_lm,
)

# %%
# Compile/Optimize the ReAct agent (use the friendlier tool for better LM usability)
optimized_diabetes_agent = teleprompter.compile(student=diabetes_agent, trainset=trainset_diabetes, valset=devset_diabetes)

# %%
optimized_diabetes_agent


# %%
# Access the detailed results from your optimized agent
results = optimized_diabetes_agent.detailed_results

# Get all candidates and their validation scores
candidates = results.candidates
val_scores = results.val_aggregate_scores

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
copd_baseline_eval = evaluator_copd(copd_agent, metric=llm_metric_prediction)

# %%
copd_baseline_eval

# %%
copd_agent.copd_agent.extract._compiled = True
copd_agent.copd_agent.react._compiled = False
teleprompter = GEPA(
    metric=llm_metric_prediction,
    max_full_evals=2,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    add_format_failure_as_feedback=True,
    reflection_lm=teacher_lm,
)

optimized_copd_agent = teleprompter.compile(student=copd_agent, trainset=trainset_copd, valset=devset_copd)

# %%
optimized_copd_agent

# %%
# Access the detailed results from your optimized agent
results = optimized_copd_agent.detailed_results

# Get all candidates and their validation scores
candidates = results.candidates

val_scores = results.val_aggregate_scores

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
print("\\n\\nEvaluating the optimized ReAct agent...")
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
lead_react


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
baseline_lead_eval = evaluator_joint(lead_react, metric=llm_metric_prediction)

# %%
baseline_lead_eval

# %%
lead_react.lead_react.extract._compiled = True
lead_react.lead_react.react._compiled = False

teleprompter_joint = GEPA(
    metric=llm_metric_prediction,
    max_full_evals=3,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    add_format_failure_as_feedback=True,
    reflection_lm=teacher_lm,
)


optimized_lead_react = teleprompter_joint.compile(student=lead_react, trainset=trainset_joint, valset=devset_joint)

# %%

# Access the detailed results from your optimized agent
results = optimized_lead_react.detailed_results

# Get all candidates and their validation scores
candidates = results.candidates

val_scores = results.val_aggregate_scores

# Find the best candidate by validation score
best_idx = results.best_idx  # This is automatically calculated
best_score = val_scores[best_idx]
best_candidate = results.best_candidate

print(f"Best candidate index: {best_idx}")
print(f"Best validation score: {best_score}")
print(f"Best candidate components: {best_candidate}")

# %%
print("\n\nEvaluating the optimized Lead ReAct agent...")
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

