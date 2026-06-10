use serde::de::DeserializeOwned;
use serde::Deserialize;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
    "i", "you", "we", "they", "this", "these", "those", "but", "or", "not", "can",
    "could", "should", "would", "have", "had", "do", "does", "did", "get", "got",
];

fn remove_stopwords(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .filter(|token| !STOPWORDS.contains(&token.as_str()))
        .cloned()
        .collect()
}

fn tokenize(text: &str, remove_stops: bool) -> Vec<String> {
    static WORD_RE: OnceLock<Regex> = OnceLock::new();
    let word_re = WORD_RE.get_or_init(|| Regex::new(r"\b\w+\b").expect("valid tokenizer regex"));

    let lowercase_text = text.to_lowercase();
    let tokens: Vec<String> = word_re
        .find_iter(&lowercase_text)
        .map(|m| m.as_str().to_string())
        .collect();

    if remove_stops {
        remove_stopwords(&tokens)
    } else {
        tokens
    }
}

#[derive(Debug, Deserialize)]
struct IndexedDocument {
    id: usize,
    title: String,
    content: String,
    tokens: Vec<String>,
    raw_tf: HashMap<String, usize>,
    tf: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct IndexData {
    inverted_index: HashMap<String, Vec<usize>>,
    document_frequency: HashMap<String, usize>,
    total_docs: usize,
    documents: Vec<IndexedDocument>,
    last_updated: f64,
}

fn compute_bm25_score(
    query_words: &[String],
    doc_tf: &HashMap<String, usize>,
    df: &HashMap<String, usize>,
    total_docs: usize,
    documents: &[IndexedDocument],
    k1: f64,
    b: f64,
) -> f64 {
    if total_docs == 0 || documents.is_empty() {
        return 0.0;
    }

    let doc_length: usize = doc_tf.values().sum();
    let total_length: usize = documents
        .iter()
        .map(|doc| doc.raw_tf.values().sum::<usize>())
        .sum();

    if total_length == 0 {
        return 0.0;
    }

    let avg_doc_length = total_length as f64 / documents.len() as f64;
    let mut score = 0.0;

    for word in query_words {
        if let (Some(&tf), Some(&df_word)) = (doc_tf.get(word), df.get(word)) {
            let tf_f = tf as f64;
            let df_f = df_word as f64;
            let total_docs_f = total_docs as f64;

            let idf = ((total_docs_f - df_f + 0.5) / (df_f + 0.5)).ln();
            let denom = tf_f + k1 * (1.0 - b + b * (doc_length as f64 / avg_doc_length));
            score += idf * (tf_f * (k1 + 1.0)) / denom;
        }
    }

    score
}

fn query_documents_with_index(query_text: &str, index_data: &IndexData) -> Vec<(usize, f64)> {
    let query_words = tokenize(query_text, true);
    if query_words.is_empty() {
        return Vec::new();
    }

    let mut candidate_doc_ids: HashSet<usize> = HashSet::new();
    for word in &query_words {
        if let Some(doc_ids) = index_data.inverted_index.get(word) {
            candidate_doc_ids.extend(doc_ids.iter().copied());
        }
    }

    if candidate_doc_ids.is_empty() {
        return Vec::new();
    }

    let mut results: Vec<(usize, f64)> = Vec::new();
    for doc_id in candidate_doc_ids {
        if let Some(doc) = index_data.documents.get(doc_id) {
            if doc.tokens.is_empty() {
                continue;
            }

            let bm25_score = compute_bm25_score(
                &query_words,
                &doc.raw_tf,
                &index_data.document_frequency,
                index_data.total_docs,
                &index_data.documents,
                1.2,
                0.75,
            );
            results.push((doc_id, bm25_score));
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn load_index_from_disk<T: DeserializeOwned>(
    index_path: &str,
) -> Result<T, Box<dyn std::error::Error>> {
    // Check if file exists
    if !Path::new(index_path).exists() {
        println!("Index file {} not found", index_path);
        return Err(format!("Index file {} not found", index_path).into());
    }

    // Read file contents
    let file_contents = fs::read_to_string(index_path)?;
    let index_data: T = serde_json::from_str(&file_contents)?;

    Ok(index_data)
}

fn main() {
    let index_data = match load_index_from_disk::<IndexData>("index.json") {
        Ok(index_data) => {
            println!(
                "Index loaded from index.json: total_docs={}, documents={}, terms={}",
                index_data.total_docs,
                index_data.documents.len(),
                index_data.inverted_index.len()
            );
            index_data
        }
        Err(err) => {
            eprintln!("Failed to load index.json: {}", err);
            return;
        }
    };
    let args: Vec<String> = std::env::args().collect();
    let command = args[1..].join(" ").trim().to_string();

    if let Some(query_text) = command.strip_prefix("query ") {
        // Stubbed to match the Python flow until the Rust search functions are ready.
        // let results = query_documents_with_index(query_text);
        let query_words = tokenize(query_text, true);
        // format_search_results(&results, &query_words);

        println!("Query stub parsed: {query_text}");
        println!("Tokenized into {} query terms", query_words.len());
    } else {
        println!("Unknown command: {command}");
        println!("Usage:");
        println!("  cargo run -- query <search terms>");
    }
}
