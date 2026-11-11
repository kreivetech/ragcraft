use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a document stored within the vector store.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct StoredDocument {
    /// Unique identifier for the document or chunk
    pub id: String,
    /// Path or source URI of the document
    pub path: String,
    /// Text content of the document or chunk
    pub content: String,
    /// Document format or type (e.g., "code", "markdown", "text")
    pub doc_type: String,
    /// Optional metadata associated with the document
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl StoredDocument {
    /// Creates a new stored document without metadata.
    pub fn new(
        id: impl Into<String>,
        path: impl Into<String>,
        content: impl Into<String>,
        doc_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            content: content.into(),
            doc_type: doc_type.into(),
            metadata: HashMap::new(),
        }
    }

    /// Attaches metadata to the stored document.
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// The result of a semantic vector search.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct RetrievalResult {
    /// Retrieved document contents ordered by relevance descending
    pub documents: Vec<String>,
    /// Corresponding relevance / cosine similarity scores
    pub scores: Vec<f32>,
}

impl RetrievalResult {
    /// Returns true if no documents were retrieved.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Returns the number of retrieved documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }
}
