import sys
import re
from collections import Counter
import math

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

    for doc in documents:
        if len(doc["tokens"]) == 0:
            continue

        # Calculate TF-IDF score
        tfidf_score = 0
        for word in query_words:
            tf = doc["tf"].get(word, 0)
            if word in df and tf > 0:
                idf = math.log(total_docs / df[word])
                tfidf_score += tf * idf

        if tfidf_score > 0:
            results.append((doc, tfidf_score, tfidf_score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

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
            results = query_documents(query_text)
            if results:
                for doc, count, percentage in results:
                    print(f"ID {doc['id']}: '{doc['title']}' (TF-IDF score: {count:.4f})")
            else:
                print("No documents found matching your query.")
        elif command == "exit":
            break
        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py <add> <title> <content>")
