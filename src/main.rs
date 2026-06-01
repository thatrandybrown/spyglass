use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

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
    match load_index_from_disk::<HashMap<String, Value>>("index.json") {
        Ok(index_data) => {
            let total_docs = index_data
                .get("total_docs")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let document_count = index_data
                .get("documents")
                .and_then(|v| v.as_array())
                .map(|docs| docs.len())
                .unwrap_or(0);

            println!(
                "Index loaded from index.json: total_docs={}, documents={}, top_level_keys={}",
                total_docs,
                document_count,
                index_data.len()
            );
        }
        Err(err) => {
            eprintln!("Failed to load index.json: {}", err);
        }
    }

    println!("Hello, world!");
}
