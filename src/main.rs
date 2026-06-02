use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

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

    println!("Hello, world!");
}
