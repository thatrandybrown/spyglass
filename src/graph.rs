use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub aliases: Vec<String>,
}
