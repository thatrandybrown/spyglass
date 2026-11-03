use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub const ENTITY_KINDS: [&str; 5] = ["person", "place", "organization", "concept", "thing"];

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
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

    pub fn add_entity(&mut self, kind: &str, name: &str) -> bool {
        let id = slugify(name);
        if id.is_empty()
            || !ENTITY_KINDS.contains(&kind)
            || self.entities.iter().any(|e| e.id == id)
        {
            return false;
        }

        self.entities.push(Entity {
            id,
            name: name.to_string(),
            kind: kind.to_string(),
            aliases: Vec::new(),
        });
        true
    }
}

pub fn load_graph(path: &str) -> Result<Graph, Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Ok(Graph::new());
    }

    let file_contents = fs::read_to_string(path)?;
    let graph: Graph = serde_json::from_str(&file_contents)?;

    Ok(graph)
}

pub fn save_graph(graph: &Graph, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(graph)?;
    fs::write(path, json)?;

    Ok(())
}

pub fn slugify(name: &str) -> String {
    name.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect::<Vec<&str>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_entity_rejects_duplicates_and_unknown_kinds() {
        let mut graph = Graph::new();

        assert!(graph.add_entity("person", "Prince Lucas"));
        assert_eq!(graph.entities[0].id, "prince-lucas");
        assert!(!graph.add_entity("place", "prince lucas"));
        assert!(!graph.add_entity("spaceship", "Someone Else"));
        assert_eq!(graph.entities.len(), 1);
    }

    #[test]
    fn add_entity_rejects_names_with_empty_slugs() {
        let mut graph = Graph::new();

        assert!(!graph.add_entity("person", "..."));
        assert!(!graph.add_entity("person", ""));
        assert!(graph.entities.is_empty());
    }

    #[test]
    fn slugify_handles_spaces_case_and_punctuation() {
        assert_eq!(slugify("Prince Lucas"), "prince-lucas");
        assert_eq!(slugify("  Prince   LUCAS  "), "prince-lucas");
        assert_eq!(slugify("St. John's, Inc."), "st-john-s-inc");
    }

    #[test]
    fn save_then_load_graph_is_equal() {
        let path = std::env::temp_dir().join("trb_spyglass_save_graph_test.json");
        let path = path.to_str().unwrap();

        let mut graph = Graph::new();
        graph.entities.push(Entity {
            id: "prince-lucas".to_string(),
            name: "Prince Lucas".to_string(),
            kind: "person".to_string(),
            aliases: vec!["Lucas".to_string()],
        });

        save_graph(&graph, path).unwrap();
        let loaded = load_graph(path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(loaded, graph);
    }

    #[test]
    fn load_graph_missing_path_is_empty() {
        let graph = load_graph("does-not-exist.json").unwrap();

        assert_eq!(graph.version, 1);
        assert!(graph.entities.is_empty());
    }

    #[test]
    fn graph_round_trip() {
        let mut graph = Graph::new();
        graph.entities.push(Entity {
            id: "prince-lucas".to_string(),
            name: "Prince Lucas".to_string(),
            kind: "person".to_string(),
            aliases: vec!["Lucas".to_string()],
        });

        let json = serde_json::to_string(&graph).unwrap();
        let loaded: Graph = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.entities.len(), 1);
        assert_eq!(loaded.entities[0].id, "prince-lucas");
        assert_eq!(loaded.entities[0].name, "Prince Lucas");
        assert_eq!(loaded.entities[0].kind, "person");
        assert_eq!(loaded.entities[0].aliases, vec!["Lucas"]);
    }
}
