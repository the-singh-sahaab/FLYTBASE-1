from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = (
    "Frame 1: A bottle of water is placed on a table. "
    "Frame 2: A refrigerator stands against the wall."
)

def test_bottle_present():
    answer = qa(question="Is there a bottle?", context=context)
    assert "yes" in answer["answer"].lower() or "bottle" in answer["answer"].lower()

def test_refrigerator_present():
    answer = qa(question="Was there a refrigerator?", context=context)
    assert "refrigerator" in answer["answer"].lower()
