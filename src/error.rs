use thiserror::Error;

/// Error types for `ragcraft`.
#[derive(Debug, Error)]
pub enum RagError {
    /// I/O operations failure
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization / deserialization failure
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Query vector is empty
    #[error("Query vector cannot be empty")]
    EmptyQuery,

    /// Vector dimension mismatch
    #[error("Vector dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    /// Document or item not found
    #[error("Document not found: {0}")]
    NotFound(String),
}

impl From<postcard::Error> for RagError {
    fn from(err: postcard::Error) -> Self {
        RagError::Serialization(format!("Postcard error: {err}"))
    }
}

impl From<serde_json::Error> for RagError {
    fn from(err: serde_json::Error) -> Self {
        RagError::Serialization(format!("JSON error: {err}"))
    }
}
