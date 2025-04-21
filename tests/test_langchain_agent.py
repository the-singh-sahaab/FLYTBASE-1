import json
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

@pytest.fixture(scope="module")
def docs():
    # Manually load JSONL as LangChain Documents
    docs = []
    file_path = Path(__file__).parent / "sample_index.jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            docs.append(Document(page_content=entry["description"], metadata=entry))
    return docs

@pytest.fixture(scope="module")
def retriever(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

@pytest.fixture(scope="module")
def qa_chain(retriever):
    pipe = pipeline("text-generation", model="gpt2", max_new_tokens=32, pad_token_id=50256)
    llm = HuggingFacePipeline(pipeline=pipe)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def test_retrieve_bottle_docs(retriever):
    docs = retriever.get_relevant_documents("bottle")
    assert any("bottle" in d.page_content.lower() for d in docs)

def test_langchain_answer_bottle(qa_chain):
    answer = qa_chain.run("Is there a bottle?")
    assert "bottle" in answer.lower()

def test_retrieve_refrigerator_docs(retriever):
    docs = retriever.get_relevant_documents("refrigerator")
    assert any("refrigerator" in d.page_content.lower() for d in docs)

def test_langchain_answer_refrigerator(qa_chain):
    answer = qa_chain.run("Was there a refrigerator?")
    assert "refrigerator" in answer.lower()
