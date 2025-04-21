from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Initialize GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2 = gpt2.to(device)

def generate_description(objects):
    """
    Generate a natural-language scene description from a list of detected objects using GPT-2.
    - objects: list of str (e.g., ["bottle", "table"])
    Returns: str (one-sentence description)
    """
    prompt = f"Describe a scene that includes: {', '.join(objects)}."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = gpt2.generate(
        input_ids,
        max_length=input_ids.shape[1] + 20,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    caption = generated[len(prompt):].strip().split("\n")[0]
    if not caption or caption.lower().startswith("describe"):
        caption = "Detected " + ", ".join(objects) + " in the scene."
    return caption
