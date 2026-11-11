//! # ragcraft
//!
//! Fast semantic retrieval, document chunking, vector metrics, and embedded vector store for AI agents and RAG applications.
//!
//! Developed and maintained by [Kreive Tech](https://kreive.com.br).
//!
//! ## Features
//!
//! - **Fast Embedded Vector Store**: In-memory vector database with $O(N \log K)$ binary heap cosine similarity search and atomic binary serialization (`postcard`).
//! - **Smart Document & Code Chunker**: Recursive and Markdown-aware document splitting with header hierarchy and token budgets.
//! - **Vector Math & Context Assembly**: Cosine similarity, dot product, L2 normalization, and automated LLM prompt context assembler.
//! - **Agnostic & Lightweight**: Zero heavy C++ bindings or PyTorch runtime required by default.
//!
//! ## Quick Start
//!
//! ```rust
//! use ragcraft::{VectorStore, StoredDocument, assemble_context};
//!
//! let mut store = VectorStore::new();
//!
//! // Add documents with their embedding vectors
//! store.add(
//!     StoredDocument::new("doc-1", "docs/auth.md", "Authentication uses JWT tokens.", "markdown"),
//!     vec![0.1, 0.8, 0.3],
//! );
//! store.add(
//!     StoredDocument::new("doc-2", "docs/db.md", "Database uses PostgreSQL connection pool.", "markdown"),
//!     vec![0.7, 0.2, 0.1],
//! );
//!
//! // Perform semantic search
//! let results = store.search(&[0.1, 0.75, 0.25], 2).unwrap();
//! assert_eq!(results.documents.len(), 2);
//!
//! // Format context block for an LLM prompt
//! let prompt_context = assemble_context(&results);
//! assert!(prompt_context.contains("Authentication uses JWT tokens"));
//! ```

pub mod chunker;
pub mod error;
pub mod metrics;
pub mod store;
pub mod types;

pub use chunker::{Chunk, Chunker};
pub use error::RagError;
pub use metrics::{
    assemble_context, assemble_context_with_threshold, cosine_sim, dot_product, l2_norm,
    l2_normalize,
};
pub use store::VectorStore;
pub use types::{RetrievalResult, StoredDocument};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e2e_rag_workflow() {
        let chunker = Chunker::new(None);
        let markdown = "# Title\nSection 1 text\n## Sub\nSection 2 text";
        let chunks = chunker.chunk_document(markdown, "guide", 50, 0, "md", &std::collections::HashMap::new());
        assert_eq!(chunks.len(), 2);

        let mut store = VectorStore::new();
        store.add(
            StoredDocument::new(&chunks[0].id, "guide.md", &chunks[0].content, "markdown"),
            vec![1.0, 0.0, 0.0],
        );
        store.add(
            StoredDocument::new(&chunks[1].id, "guide.md", &chunks[1].content, "markdown"),
            vec![0.0, 1.0, 0.0],
        );

        let query = vec![0.9, 0.1, 0.0];
        let results = store.search(&query, 1).unwrap();
        assert_eq!(results.documents.len(), 1);
        assert!(results.documents[0].contains("Section 1 text"));

        let context = assemble_context(&results);
        assert!(context.contains("RELEVANT CONTEXT"));
    }
}
