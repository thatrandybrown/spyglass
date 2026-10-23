use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Graph {
    pub version: u32,
    pub entities: Vec<Entity>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            version: 1,
            entities: Vec::new(),
        }
    }
}
