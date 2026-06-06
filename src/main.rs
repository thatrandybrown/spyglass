use serde::de::DeserializeOwned;
use serde::Deserialize;
use regex::Regex;
use std::collections::HashMap;
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
    match load_index_from_disk::<IndexData>("index.json") {
        Ok(index_data) => {
            println!(
                "Index loaded from index.json: total_docs={}, documents={}, terms={}",
                index_data.total_docs,
                index_data.documents.len(),
                index_data.inverted_index.len()
            );
        }
        Err(err) => {
            eprintln!("Failed to load index.json: {}", err);
        }
    }
    let args: Vec<String> = std::env::args().collect();

    println!("Hello, world!");
}
