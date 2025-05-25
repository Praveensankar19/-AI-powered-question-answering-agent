import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import numpy as np
import sys
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

# 1. Web Crawler -----------------------
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

def crawl_website(base_url, max_pages=10):
    visited = set()
    to_visit = [base_url]
    to_visit_set = {base_url}
    crawled_data = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        to_visit_set.remove(current_url)

        if current_url in visited:
            continue
        try:
            response = requests.get(current_url, timeout=5)
            if response.status_code != 200:
                continue
        except Exception as e:
            print(f"Error fetching {current_url}: {e}")
            continue

        print(f"Crawling: {current_url}")
        visited.add(current_url)
        page_text = extract_text_from_html(response.text)

        if len(page_text) > 100:
            crawled_data.append({'url': current_url, 'text': page_text})

        soup = BeautifulSoup(response.text, 'html.parser')
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            full_url = urljoin(current_url, href)
            if base_url in full_url and full_url not in visited and full_url not in to_visit_set:
                to_visit.append(full_url)
                to_visit_set.add(full_url)

        time.sleep(1)

    return crawled_data

# 2. Text Splitter -------------------------
def split_text(text, max_len=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for word in words:
        current_chunk.append(word)
        current_len += len(word) + 1
        if current_len >= max_len:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_len = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# 3. Embeddings -------------------------
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return EMBEDDING_MODEL.encode(text, convert_to_tensor=True)

# 4. Cosine Similarity -------------------------
def cosine_similarity(vec1, vec2):
    vec1 = vec1.cpu().numpy() if torch.is_tensor(vec1) else np.array(vec1)
    vec2 = vec2.cpu().numpy() if torch.is_tensor(vec2) else np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 5. Build Vector DB -------------------------
def build_vector_db(docs):
    vector_db = []
    for doc in docs:
        chunks = split_text(doc['text'])
        for chunk in chunks:
            emb = get_embedding(chunk)
            vector_db.append({'embedding': emb, 'text': chunk, 'url': doc['url']})
    return vector_db

# 6. Vector Search -------------------------
def search_similar_chunks(query, vector_db, top_k=3, min_score=0.1):
    query_emb = get_embedding(query)
    similarities = []
    for item in vector_db:
        score = cosine_similarity(query_emb, item['embedding'])
        if score >= min_score:
            similarities.append((score, item))
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in similarities[:top_k]]

# 7. Answer Generator -------------------------
roberta_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
roberta_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def generate_answer(question, relevant_chunks):
    if not relevant_chunks:
        return "Sorry, I couldn't find any information about that in the documentation."

    best_answer = ""
    best_score = float("-inf")
    best_url = ""

    for chunk in relevant_chunks:
        context = chunk["text"]
        inputs = roberta_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = roberta_model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)

        if start_idx <= end_idx:
            answer_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
            answer_tokens = roberta_tokenizer.convert_ids_to_tokens(answer_ids)
            answer = roberta_tokenizer.convert_tokens_to_string(answer_tokens).strip()

            if answer.lower() in ["", "<s>", "</s>", "[cls]", "[sep]"]:
                continue

            score = start_scores[0][start_idx] + end_scores[0][end_idx]

            if score > best_score:
                best_answer = answer
                best_score = score
                best_url = chunk["url"]

    if best_answer:
        return f"{best_answer}\n(Source: {best_url})"

    # === Fallback to T5 ===
    best_chunk = relevant_chunks[0]
    prompt = f"question: {question} context: {best_chunk['text']}"
    input_ids = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids

    with torch.no_grad():
        outputs = t5_model.generate(input_ids, max_length=100, num_beams=2)

    fallback_answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if fallback_answer == "" or fallback_answer.lower() in ["<s>", "none"]:
        fallback_answer = "Sorry, I couldn't find a clear answer in the documentation."

    return f"{fallback_answer}\n(Source: {best_chunk['url']})"

# 8. CLI Interface -------------------------
def main(url):
    print("Starting crawling...")
    docs = crawl_website(url, max_pages=10)
    print(f"Crawled {len(docs)} pages.")

    print("Building vector database...")
    vector_db = build_vector_db(docs)
    print(f"Built vector DB with {len(vector_db)} chunks.")

    print("You can now ask questions about the documentation. Type 'exit' to quit.")
    while True:
        query = input("> ")
        if query.lower() in ['exit', 'quit']:
            print("Bye! Have a nice day.")
            break
        relevant_chunks = search_similar_chunks(query, vector_db)
        answer = generate_answer(query, relevant_chunks)
        print("Answer:", answer)

# Entry Point -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python qa_agent.py https://help.example.com")
        sys.exit(1)
    start_url = sys.argv[1]
    if not is_valid_url(start_url):
        print("Invalid URL. Please provide a valid help website URL.")
        sys.exit(1)
    main(start_url)
