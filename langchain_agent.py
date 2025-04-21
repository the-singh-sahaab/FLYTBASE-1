import json
from datetime import datetime
from transformers import pipeline
from ctransformers import AutoModelForCausalLM

def load_session_logs(session_log_path):
    """Load a session's JSONL log file into a list of dicts."""
    logs = []
    with open(session_log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except Exception:
                continue
    return logs

def human_time(ts):
    """Convert UNIX timestamp to human-readable time."""
    return datetime.fromtimestamp(ts).strftime('%I:%M %p')

def filter_logs_by_keyword(logs, keyword):
    """Return logs whose description contains the keyword (case-insensitive)."""
    keyword = keyword.lower()
    return [log for log in logs if keyword in log.get("description", "").lower()]

def format_events(events):
    """Format events as a list of lines with time, description, and video."""
    lines = []
    for log in events:
        tstr = human_time(log["timestamp"])
        desc = log.get("description", "")
        vpath = log.get("video_path", "")
        lines.append(f"{tstr}: {desc} [Video: {vpath}]")
    return lines

# Initialize Phi-2 model for summarization
phi_model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/phi-2-GGUF",
    model_file="models/phi-2.Q4_K_M.gguf",
    model_type="phi"
)

def summarize_events(events, keyword, max_events=10):
    """
    Use Phi-2 to summarize the filtered events with a fixed prompt.
    Truncates the event list if it would exceed the model's max input length.
    """
    if not events:
        return f"No events found related to '{keyword}'."
    events = events[:max_events]
    event_lines = format_events(events)
    prompt = (
        f"Summarize the following events related to '{keyword}', listing them with time and description:\n"
        + "\n".join(event_lines)
    )
    # Truncate prompt if too long for the model (e.g., 2048 tokens ≈ 8000 chars for Phi-2)
    max_prompt_len = 7000
    if len(prompt) > max_prompt_len:
        prompt = prompt[:max_prompt_len]
    result = phi_model(prompt, max_new_tokens=100)
    return result.strip()

def main():
    print("=== LangChain Session QA Agent ===")
    session_log_path = input("Enter path to session log (e.g., recent_indexes/index_run_007.jsonl): ").strip()
    if not session_log_path or not session_log_path.endswith(".jsonl"):
        print("Please provide a valid session .jsonl file.")
        return
    logs = load_session_logs(session_log_path)
    # Force LLM to run on CPU to avoid CUDA errors
    llm = pipeline("text-generation", model="gpt2", device=-1, max_new_tokens=128, pad_token_id=50256)
    while True:
        query = input("\nEnter your event query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        # Extract keyword (for demo, use last noun or word after 'related to' or 'about')
        import re
        m = re.search(r"(?:related to|about|of|for)\s+([a-zA-Z0-9_]+)", query.lower())
        if m:
            keyword = m.group(1)
        else:
            # fallback: use last word
            keyword = query.strip().split()[-1]
        filtered = filter_logs_by_keyword(logs, keyword)
        answer = summarize_events(filtered, keyword, max_events=10)
        print("\n" + answer)

if __name__ == "__main__":
    main()
