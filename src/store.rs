use crate::error::RagError;
use crate::metrics::cosine_sim;
use crate::types::{RetrievalResult, StoredDocument};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::path::Path;

/// In-memory vector store with fast top-k cosine similarity search and binary persistence.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VectorStore {
    /// Mapping of document ID to vector embedding
    pub embeddings: HashMap<String, Vec<f32>>,
    /// Mapping of document ID to document payload
    pub documents: HashMap<String, StoredDocument>,
}

#[derive(PartialEq)]
struct ScoredEntry(f32, String);

impl Eq for ScoredEntry {}

impl Ord for ScoredEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for ScoredEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl VectorStore {
    /// Creates an empty vector store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a single document with its embedding vector.
    pub fn add(&mut self, doc: StoredDocument, embedding: Vec<f32>) {
        if embedding.is_empty() {
            return;
        }
        self.embeddings.insert(doc.id.clone(), embedding);
        self.documents.insert(doc.id.clone(), doc);
    }

    /// Adds a batch of documents and their corresponding embedding vectors.
    pub fn add_batch(&mut self, docs: Vec<(StoredDocument, Vec<f32>)>) {
        for (doc, emb) in docs {
            self.add(doc, emb);
        }
    }

    /// Searches for the top-k most similar documents to the query vector.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<RetrievalResult, RagError> {
        if query.is_empty() {
            return Err(RagError::EmptyQuery);
        }
        if self.embeddings.is_empty() || top_k == 0 {
            return Ok(RetrievalResult::default());
        }

        let mut heap: BinaryHeap<std::cmp::Reverse<ScoredEntry>> =
            BinaryHeap::with_capacity(top_k + 1);

        for (id, emb) in &self.embeddings {
            let score = cosine_sim(query, emb).max(0.0);
            let entry = std::cmp::Reverse(ScoredEntry(score, id.clone()));
            if heap.len() < top_k {
                heap.push(entry);
            } else if let Some(worst) = heap.peek() {
                if score.total_cmp(&worst.0.0) == std::cmp::Ordering::Greater {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }

        let mut top: Vec<ScoredEntry> = heap.into_iter().map(|r| r.0).collect();
        top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        let mut documents = Vec::with_capacity(top.len());
        let mut scores = Vec::with_capacity(top.len());

        for ScoredEntry(score, id) in top {
            if let Some(doc) = self.documents.get(&id) {
                documents.push(doc.content.clone());
                scores.push(score);
            }
        }

        Ok(RetrievalResult { documents, scores })
    }

    /// Removes a document and its embedding by ID.
    pub fn remove(&mut self, id: &str) -> Option<StoredDocument> {
        self.embeddings.remove(id);
        self.documents.remove(id)
    }

    /// Retrieves a document reference by ID.
    pub fn get(&self, id: &str) -> Option<&StoredDocument> {
        self.documents.get(id)
    }

    /// Retrieves an embedding vector reference by document ID.
    pub fn get_embedding(&self, id: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(id)
    }

    /// Returns the number of stored documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Returns true if the store contains no documents.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Clears all documents and embeddings from memory.
    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.documents.clear();
    }

    /// Returns the document and embedding counts.
    pub fn stats(&self) -> (usize, usize) {
        (self.documents.len(), self.embeddings.len())
    }

    /// Saves the vector store atomically to disk in compact postcard binary format.
    pub fn save_to_disk(&self, path: &Path) -> Result<(), RagError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = postcard::to_stdvec(self)?;
        let tmp_path = format!("{}.tmp", path.display());
        fs::write(&tmp_path, bytes)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Loads a vector store from a binary file on disk (with fallback to legacy JSON if available).
    pub fn load_from_disk(path: &Path) -> Result<Self, RagError> {
        if path.exists() {
            let bytes = fs::read(path)?;
            let store: Self = postcard::from_bytes(&bytes)?;
            return Ok(store);
        }

        // Try JSON fallback
        let json_path = path.with_extension("json");
        if json_path.exists() {
            let content = fs::read_to_string(&json_path)?;
            let store: Self = serde_json::from_str(&content)?;
            return Ok(store);
        }

        Err(RagError::NotFound(path.display().to_string()))
    }

    /// Loads an existing vector store or creates a new empty one if the file does not exist.
    pub fn load_or_create(path: &Path) -> Result<Self, RagError> {
        match Self::load_from_disk(path) {
            Ok(store) => Ok(store),
            Err(RagError::NotFound(_)) => Ok(Self::new()),
            Err(err) => Err(err),
        }
    }
}

#[cfg(feature = "async")]
impl VectorStore {
    /// Saves the vector store asynchronously to disk in compact binary format.
    pub async fn save_to_disk_async(&self, path: &Path) -> Result<(), RagError> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let bytes = postcard::to_stdvec(self)?;
        let tmp_path = format!("{}.tmp", path.display());
        tokio::fs::write(&tmp_path, bytes).await?;
        tokio::fs::rename(&tmp_path, path).await?;
        Ok(())
    }

    /// Loads a vector store asynchronously from disk.
    pub async fn load_from_disk_async(path: &Path) -> Result<Self, RagError> {
        if path.exists() {
            let bytes = tokio::fs::read(path).await?;
            let store: Self = postcard::from_bytes(&bytes)?;
            return Ok(store);
        }

        let json_path = path.with_extension("json");
        if json_path.exists() {
            let content = tokio::fs::read_to_string(&json_path).await?;
            let store: Self = serde_json::from_str(&content)?;
            return Ok(store);
        }

        Err(RagError::NotFound(path.display().to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_doc(id: &str, content: &str) -> StoredDocument {
        StoredDocument::new(id, id, content, "test")
    }

    fn unit(dim: usize, idx: usize, scale: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx] = scale;
        v
    }

    #[test]
    fn search_empty_store_returns_empty() {
        let store = VectorStore::new();
        let result = store.search(&unit(4, 0, 1.0), 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn search_empty_query_returns_error() {
        let store = VectorStore::new();
        let err = store.search(&[], 5).unwrap_err();
        assert!(matches!(err, RagError::EmptyQuery));
    }

    #[test]
    fn add_empty_embedding_skipped() {
        let mut store = VectorStore::new();
        let doc = make_doc("doc-empty", "content");
        store.add(doc, vec![]);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn search_returns_results_in_descending_score_order() {
        let mut store = VectorStore::new();
        store.add(make_doc("doc-a", "alpha"), unit(4, 0, 1.0));
        store.add(make_doc("doc-b", "beta"), unit(4, 1, 1.0));
        store.add(make_doc("doc-c", "gamma"), unit(4, 2, 1.0));

        let results = store.search(&unit(4, 2, 1.0), 3).unwrap();
        assert_eq!(results.documents.len(), 3);
        assert!(results.scores[0] >= results.scores[1]);
        assert!(results.scores[1] >= results.scores[2]);
        assert_eq!(results.documents[0], "gamma");
    }

    #[test]
    fn save_and_load_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut store = VectorStore::new();
        store.add(make_doc("d1", "Rust is fast"), vec![0.1, 0.2, 0.3]);
        store.add(make_doc("d2", "Python is simple"), vec![0.4, 0.5, 0.6]);

        store.save_to_disk(&path).unwrap();
        assert!(path.exists());

        let loaded = VectorStore::load_from_disk(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("d1").unwrap().content, "Rust is fast");
    }
}
