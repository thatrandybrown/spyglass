import sys
import re
from collections import Counter
import math
import json
import os
import time

# Document storage - simple in-memory list of dictionaries
documents = []

def tokenize(text):
    """Split text into lowercase tokens, removing punctuation"""
    # Convert to lowercase and split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def compute_term_frequency(tokens):
    """Compute term frequency for a list of tokens"""
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {}

    term_counts = Counter(tokens)
    # Convert to relative frequencies (TF = count / total_tokens)
    tf = {term: count / total_tokens for term, count in term_counts.items()}
    return tf

def compute_document_frequency():
    """Compute how many documents each term appears in"""
    df = {}
    for doc in documents:
        unique_terms = set(doc["tokens"])
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1
    return df

def compute_query_vector(query_words, df, total_docs):
    """Convert query to TF-IDF vector"""
    query_tf = Counter(query_words)
    query_vector = {}
    for word in query_tf:
        tf = query_tf[word] / len(query_words)
        if word in df:
            idf = math.log(total_docs / df[word])
            query_vector[word] = tf * idf
    return query_vector

def compute_cosine_similarity(query_vector, doc_tf, df, total_docs):
    """Compute cosine similarity between query and document vectors"""
    dot_product = 0
    query_norm = 0
    doc_norm = 0

    # Calculate dot product and norms
    for word, query_weight in query_vector.items():
        doc_tf_val = doc_tf.get(word, 0)
        if doc_tf_val > 0:
            doc_idf = math.log(total_docs / df[word])
            doc_weight = doc_tf_val * doc_idf
            dot_product += query_weight * doc_weight
            doc_norm += doc_weight ** 2
        query_norm += query_weight ** 2

    # Add document terms not in query to doc_norm
    for word, tf_val in doc_tf.items():
        if word not in query_vector and word in df:
            doc_idf = math.log(total_docs / df[word])
            doc_weight = tf_val * doc_idf
            doc_norm += doc_weight ** 2

    if query_norm == 0 or doc_norm == 0:
        return 0

    return dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))

def compute_bm25_score(query_words, doc_tf, df, total_docs, k1=1.2, b=0.75):
    """Compute BM25 score between query and document"""
    doc_length = sum(doc_tf.values())
    avg_doc_length = sum(sum(doc["raw_tf"].values()) for doc in documents) / len(documents)
    score = 0
    for word in query_words:
        if word in doc_tf and word in df:
            tf = doc_tf[word]
            idf = math.log((total_docs - df[word] + 0.5) / (df[word] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
    return score

def build_inverted_index():
    """Build inverted index: term -> list of doc IDs containing that term"""
    inverted_index = {}
    for doc in documents:
        for term in set(doc["tokens"]):  # unique terms only
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append(doc["id"])
    return inverted_index

# read document from file path
def read_document(file_path):
    with open(file_path, "r") as file:
        return file.read()

def is_file_already_indexed(filepath):
    """Check if a file is already in the index"""
    return any(doc["title"] == filepath for doc in documents)

# Modified add_document to check for duplicates
def add_document(title, content):
    if is_file_already_indexed(title):
        print(f"File '{title}' already indexed, skipping")
        return

    doc_id = len(documents)
    tokens = tokenize(content)
    # keep normalized for cosine similarity
    tf = compute_term_frequency(tokens)
    # raw for BM25
    raw_tf = dict(Counter(tokens))

    documents.append({
        "id": doc_id,
        "title": title,
        "content": content,
        "tokens": tokens,
        "raw_tf": raw_tf,
        "tf": tf
    })
    print(f"Added document '{title}' with ID {doc_id}")

def query_documents(query_text):
    results = []
    query_words = tokenize(query_text)
    df = compute_document_frequency()
    total_docs = len(documents)
    query_vector = compute_query_vector(query_words, df, total_docs)

    for doc in documents:
        if len(doc["tokens"]) == 0:
            continue

        # Calculate cosine similarity score
        similarity_score = compute_cosine_similarity(query_vector, doc["tf"], df, total_docs)

        if similarity_score > 0:
            results.append((doc, similarity_score, similarity_score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

def query_documents_with_index(query_text):
    """Query using inverted index for efficiency"""
    query_words = tokenize(query_text)
    if not query_words:
        return []

    inverted_index = build_inverted_index()
    candidate_doc_ids = set()

    for word in query_words:
        if word in inverted_index:
            candidate_doc_ids.update(inverted_index[word])

    if not candidate_doc_ids:
        return []

    results = []
    df = compute_document_frequency()
    total_docs = len(documents)
    query_vector = compute_query_vector(query_words, df, total_docs)

    for doc_id in candidate_doc_ids:
        doc = documents[doc_id]
        if len(doc["tokens"]) == 0:
            continue

        # similarity_score = compute_cosine_similarity(query_vector, doc["tf"], df, total_docs)
        # if similarity_score > 0:
        #     results.append((doc, similarity_score, similarity_score))
        bm25_score = compute_bm25_score(query_words, doc["raw_tf"], df, total_docs)
        results.append((doc, bm25_score, bm25_score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

def extract_snippet(content, query_words, max_sentences=2):
    """Extract sentences containing query terms"""
    import re

    query_words_lower = [word.lower() for word in query_words]

    # Split content into sentences (simple approach)
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Find sentences containing query words
    matching_sentences = []
    for sentence in sentences:
        sentence_words = [word.lower() for word in re.findall(r'\b\w+\b', sentence)]
        if any(query_word in sentence_words for query_word in query_words_lower):
            # Highlight query terms in the sentence
            highlighted_sentence = sentence
            for word in query_words:
                # Case-insensitive replacement with word boundaries
                pattern = r'\b' + re.escape(word) + r'\b'
                highlighted_sentence = re.sub(pattern, f"**{word}**", highlighted_sentence, flags=re.IGNORECASE)
            matching_sentences.append(highlighted_sentence)

            if len(matching_sentences) >= max_sentences:
                break

    if not matching_sentences:
        # Fallback: return first sentence if no matches found
        return sentences[0] if sentences else "No content available"

    return " ".join(matching_sentences)

def save_index_to_disk(index_path="index.json"):
    inverted_index = build_inverted_index()
    df = compute_document_frequency()

    index_data = {
        "inverted_index": inverted_index,
        "document_frequency": df,
        "total_docs": len(documents),
        "documents": documents,
        # "indexed_files": [doc["title"] for doc in documents],  # Track file paths
        "last_updated": time.time()
    }

    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"Index and {len(documents)} documents saved to {index_path}")

def load_index_from_disk(index_path="index.json"):
    """Load inverted index, document metadata, and full documents from disk"""
    global documents

    if not os.path.exists(index_path):
        print(f"Index file {index_path} not found")
        return None

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    # Restore the documents list
    documents = index_data["documents"]

    print(f"Index loaded: {len(documents)} documents from {index_path}")
    return index_data

if __name__ == "__main__":
    # Try to load existing index on startup
    load_index_from_disk()

    command = " ".join(sys.argv[1:]).strip()

    if command.startswith("add "):
        filepath = command.split(" ", 1)[1]
        add_document(filepath, read_document(filepath))
        print(f"Total documents: {len(documents)}")
    elif command.startswith("query "):
        query_text = command.split(" ", 1)[1]
        results = query_documents_with_index(query_text)  # Use the indexed version
        if results:
            for doc, count, percentage in results:
                query_words = tokenize(query_text)
                snippet = extract_snippet(doc["content"], query_words)
                print(f"ID {doc['id']}: '{doc['title']}' (BM25 score: {count:.4f})")
                print(f"Snippet: {snippet}\n")
        else:
            print("No documents found matching your query.")
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py <add> <title> <content>")

    if documents:
        save_index_to_disk()
        print("Index saved automatically on exit")
