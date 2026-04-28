use std::fs;
use std::path::Path;

fn load_index_from_disk(index_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Check if file exists
    if !Path::new(index_path).exists() {
        println!("Index file {} not found", index_path);
        return Err(format!("Index file {} not found", index_path).into());
    }

    // Read file contents
    let file_contents = fs::read_to_string(index_path)?;

    Ok(file_contents)
}

fn main() {
    println!("Hello, world!");
}
