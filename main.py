import sys
import re
from collections import Counter
import math
import json
import os

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

def add_document(title, content):
    doc_id = len(documents)
    tokens = tokenize(content)
    tf = compute_term_frequency(tokens)

    documents.append({
        "id": doc_id,
        "title": title,
        "content": content,
        "tokens": tokens,
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

    # Find candidate documents using inverted index
    inverted_index = build_inverted_index()
    candidate_doc_ids = set()

    for word in query_words:
        if word in inverted_index:
            candidate_doc_ids.update(inverted_index[word])

    if not candidate_doc_ids:
        return []

    # Now only compute similarity for candidate documents
    results = []
    df = compute_document_frequency()
    total_docs = len(documents)
    query_vector = compute_query_vector(query_words, df, total_docs)

    for doc_id in candidate_doc_ids:
        doc = documents[doc_id]
        if len(doc["tokens"]) == 0:
            continue

        similarity_score = compute_cosine_similarity(query_vector, doc["tf"], df, total_docs)
        if similarity_score > 0:
            results.append((doc, similarity_score, similarity_score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

def save_index_to_disk(index_path="index.json"):
    """Save inverted index and document metadata to disk"""
    inverted_index = build_inverted_index()
    df = compute_document_frequency()

    index_data = {
        "inverted_index": inverted_index,
        "document_frequency": df,
        "total_docs": len(documents),
        "documents_metadata": [
            {
                "id": doc["id"],
                "title": doc["title"],
                "tf": doc["tf"]
            } for doc in documents
        ]
    }

    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"Index saved to {index_path}")

def load_index_from_disk(index_path="index.json"):
    """Load inverted index and document metadata from disk"""
    if not os.path.exists(index_path):
        print(f"Index file {index_path} not found")
        return None

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    print(f"Index loaded from {index_path}")
    return index_data

if __name__ == "__main__":
    command = ""
    # switch on first command line argument
    while command!= "exit":
        command = input("\n> ").strip()
        print(command)
        if command.startswith("add "):
            filepath = command.split(" ", 1)[1]
            add_document(filepath, read_document(filepath))
            print(f"Total documents: {len(documents)}")
        elif command.startswith("query "):
            query_text = command.split(" ", 1)[1]
            results = query_documents_with_index(query_text)  # Use the indexed version
            if results:
                for doc, count, percentage in results:
                    print(f"ID {doc['id']}: '{doc['title']}' (Cosine similarity: {count:.4f})")
            else:
                print("No documents found matching your query.")
        elif command == "exit":
            break
        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py <add> <title> <content>")
