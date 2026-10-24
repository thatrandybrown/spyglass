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

#[cfg(test)]
mod tests {
    use super::*;

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
