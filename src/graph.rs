use regex::Regex;
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
pub struct Mention {
    pub entity_id: String,
    pub source: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct Graph {
    pub version: u32,
    pub entities: Vec<Entity>,
    #[serde(default)]
    pub mentions: Vec<Mention>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            version: 1,
            entities: Vec::new(),
            mentions: Vec::new(),
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

    pub fn add_alias(&mut self, id: &str, alias: &str) -> bool {
        let Some(entity) = self.entities.iter_mut().find(|e| e.id == id) else {
            return false;
        };
        let lowered = alias.to_lowercase();
        if entity.aliases.iter().any(|a| a.to_lowercase() == lowered) {
            return false;
        }

        entity.aliases.push(alias.to_string());
        true
    }

    pub fn find(&self, name: &str) -> Vec<&Entity> {
        let lowered = name.to_lowercase();
        self.entities
            .iter()
            .filter(|e| {
                e.name.to_lowercase() == lowered
                    || e.aliases.iter().any(|a| a.to_lowercase() == lowered)
            })
            .collect()
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

pub fn find_term_spans(text: &str, term: &str) -> Vec<(usize, usize)> {
    if slugify(term).is_empty() {
        return Vec::new();
    }

    let starts_word = term.chars().next().is_some_and(|c| c.is_alphanumeric());
    let ends_word = term.chars().last().is_some_and(|c| c.is_alphanumeric());
    let pattern = format!(
        "(?i){}{}{}",
        if starts_word { r"\b" } else { "" },
        regex::escape(term),
        if ends_word { r"\b" } else { "" }
    );
    let re = Regex::new(&pattern).unwrap();

    re.find_iter(text).map(|m| (m.start(), m.end())).collect()
}

pub fn extract_mentions(graph: &Graph, title: &str, content: &str) -> Vec<Mention> {
    let mut mentions = Vec::new();
    for entity in &graph.entities {
        for term in std::iter::once(&entity.name).chain(&entity.aliases) {
            for (start, end) in find_term_spans(content, term) {
                mentions.push(Mention {
                    entity_id: entity.id.clone(),
                    source: title.to_string(),
                    start,
                    end,
                });
            }
        }
    }

    mentions
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
    fn add_alias_rejects_unknown_ids_and_duplicates() {
        let mut graph = Graph::new();
        graph.add_entity("person", "Prince Lucas");

        assert!(graph.add_alias("prince-lucas", "Lucas"));
        assert!(!graph.add_alias("prince-lucas", "lucas"));
        assert!(!graph.add_alias("nobody", "Lucas"));
        assert_eq!(graph.entities[0].aliases, vec!["Lucas"]);
    }

    #[test]
    fn find_matches_names_and_aliases_ignoring_case() {
        let mut graph = Graph::new();
        graph.add_entity("person", "Prince Lucas");
        graph.add_entity("person", "Lucas Grey");
        graph.add_alias("prince-lucas", "Lucas");
        graph.add_alias("lucas-grey", "Lucas");

        let ids = |found: Vec<&Entity>| found.iter().map(|e| e.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(graph.find("prince lucas")), vec!["prince-lucas"]);
        assert_eq!(ids(graph.find("LUCAS")), vec!["prince-lucas", "lucas-grey"]);
        assert!(graph.find("Luc").is_empty());
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
    fn find_term_spans_returns_each_match() {
        let text = "Lucas met Rome. Then lucas left.";

        assert_eq!(find_term_spans(text, "Lucas"), vec![(0, 5), (21, 26)]);
        assert_eq!(
            find_term_spans("St. John's, Inc. was here", "Inc."),
            vec![(12, 16)]
        );
        assert!(find_term_spans(text, "").is_empty());
        assert!(find_term_spans("wait... what...", "...").is_empty());
    }

    #[test]
    fn extract_mentions_finds_names_and_aliases() {
        let mut graph = Graph::new();
        graph.add_entity("person", "Prince Lucas");
        graph.add_entity("place", "Rome");
        graph.add_alias("prince-lucas", "Lucas");

        let mentions = extract_mentions(&graph, "notes/trip.md", "Prince Lucas sailed to Rome.");
        let found: Vec<(&str, usize, usize)> = mentions
            .iter()
            .map(|m| (m.entity_id.as_str(), m.start, m.end))
            .collect();

        assert_eq!(
            found,
            vec![
                ("prince-lucas", 0, 12),
                ("prince-lucas", 7, 12),
                ("rome", 23, 27)
            ]
        );
        assert!(mentions.iter().all(|m| m.source == "notes/trip.md"));
    }

    #[test]
    fn find_term_spans_skips_matches_inside_words() {
        assert!(find_term_spans("Lucasfilm and McLucas", "Lucas").is_empty());
        assert!(find_term_spans("see Lucas_notes", "Lucas").is_empty());
    }

    #[test]
    fn find_term_spans_slices_unicode_text_cleanly() {
        let text = "Café ☕ with ZOË and Zoë";
        let spans = find_term_spans(text, "zoë");

        assert_eq!(spans.len(), 2);
        assert_eq!(&text[spans[0].0..spans[0].1], "ZOË");
        assert_eq!(&text[spans[1].0..spans[1].1], "Zoë");
    }

    #[test]
    fn graph_without_mentions_still_loads() {
        let graph: Graph = serde_json::from_str(r#"{"version":1,"entities":[]}"#).unwrap();

        assert!(graph.mentions.is_empty());
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
