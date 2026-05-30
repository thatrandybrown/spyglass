use serde_json::Value;
use std::fs;
use std::path::Path;

fn load_index_from_disk(index_path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    // Check if file exists
    if !Path::new(index_path).exists() {
        println!("Index file {} not found", index_path);
        return Err(format!("Index file {} not found", index_path).into());
    }

    // Read file contents
    let file_contents = fs::read_to_string(index_path)?;
    let index_data: Value = serde_json::from_str(&file_contents)?;

    Ok(index_data)
}

fn main() {
    println!("Hello, world!");
}
