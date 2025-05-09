import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# HF pipeline wrappers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Optional: Suppress TensorFlow warnings (e.g., cuDNN, AVX, oneDNN logs)
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain_community.llms",
)

# --------------------------------------------------------------------------- #
CHROMA_PATH = "chroma"        # directory with your persisted Chroma DB
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HF_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # pick what fits your RAM

PROMPT_TEMPLATE = """
Answer the question using only the context provided between the lines.

{context}
---
Question: {question}

Answer:"""
# --------------------------------------------------------------------------- #


def build_llm(model_name: str) -> HuggingFacePipeline:
    """Load a local HF chat-tuned model and wrap it for LangChain."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",      # GPU if available, else CPU
        torch_dtype="auto",     # float16 if GPU, else float32
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=text_gen)


def ask(query: str, top_k: int = 3) -> None:
    # 1. Embeddings -------------------------------------------------------- #
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 2. Vector DB --------------------------------------------------------- #
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)

    # 3. Retrieve chunks --------------------------------------------------- #
    results = db.similarity_search_with_relevance_scores(query, k=top_k)
    if not results:
        print("Unable to find sufficiently relevant results.")
        return
    
    context_text = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    # 4. Craft the prompt -------------------------------------------------- #
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query
    )

    # 5. Local LLM --------------------------------------------------------- #
    llm = build_llm(HF_MODEL_NAME)
    answer = llm.invoke(prompt)

    # 6. Display ----------------------------------------------------------- #
    print("─" * 80)
    print(answer.strip())
    print("\nSources:")
    for doc, _ in results:
        print(" •", doc.metadata.get("source", "unknown"))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Local Q&A over Chroma.")
    cli.add_argument("query", help="The question to ask.")
    args = cli.parse_args()

    ask(args.query)
