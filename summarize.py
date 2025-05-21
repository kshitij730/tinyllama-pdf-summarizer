import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from concurrent.futures import ThreadPoolExecutor

# Load Hugging Face TinyLlama model
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_summary(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_pdf_chunks(file, min_len=100):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []
    for page in doc:
        text = page.get_text()
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) > min_len:
                chunks.append(para)
    return chunks

def chunk_sections(chunks, size=5):
    return [chunks[i:i+size] for i in range(0, len(chunks), size)]

def summarize_chunks(chunks):
    # First-level chunk summaries
    with ThreadPoolExecutor(max_workers=4) as executor:
        summaries = list(executor.map(lambda c: generate_summary(f"Summarize:\n{c}"), chunks))

    # Mid-level summaries per section
    section_chunks = chunk_sections(summaries, 5)
    section_summaries = [generate_summary("Combine these summaries:\n" + "\n".join(sec)) for sec in section_chunks]

    # Final abstract
    final_summary = generate_summary("Write a final summary:\n" + "\n".join(section_summaries), max_new_tokens=300)
    return final_summary
