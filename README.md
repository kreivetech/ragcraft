# ragcraft

[![Crates.io](https://img.shields.io/crates/v/ragcraft.svg)](https://crates.io/crates/ragcraft)
[![Docs.rs](https://docs.rs/ragcraft/badge.svg)](https://docs.rs/ragcraft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast semantic retrieval, smart document chunker, vector metrics, and embedded vector store for AI agents and RAG applications.

Built and maintained by **[Kreive Tech](https://kreive.com.br)**.

---

## Features

- **Embedded Vector Store**: In-memory vector database with $O(N \log K)$ binary heap cosine similarity search.
- **Smart Document Chunker**: Recursive and Markdown-aware document splitting with header hierarchy and token budgets.
- **Atomic Binary Persistence**: Fast and compact disk serialization via `postcard` binary format (with automatic JSON migration fallback).
- **Vector Math & Context Assembly**: Built-in cosine similarity, dot product, L2 normalization, and automated LLM prompt context assembler.
- **Lightweight & Fast**: Zero C++ compiler dependencies or heavyweight PyTorch runtimes required.

---

## Installation

Add `ragcraft` to your `Cargo.toml`:

```toml
[dependencies]
ragcraft = "0.1.0"
```

---

## Quick Examples

### 1. Semantic Search and LLM Context Assembly

```rust
use ragcraft::{assemble_context, StoredDocument, VectorStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut store = VectorStore::new();

    // Index documents with embeddings
    store.add(
        StoredDocument::new("doc-1", "docs/auth.md", "Authentication uses JWT tokens.", "markdown"),
        vec![0.1, 0.8, 0.3],
    );
    store.add(
        StoredDocument::new("doc-2", "docs/db.md", "Database uses PostgreSQL connection pool.", "markdown"),
        vec![0.7, 0.2, 0.1],
    );

    // Query with embedding vector
    let query_vector = vec![0.1, 0.75, 0.25];
    let results = store.search(&query_vector, 2)?;

    // Format retrieved documents into an LLM context block
    let context_block = assemble_context(&results);
    println!("{}", context_block);

    Ok(())
}
```

---

### 2. Document & Code Chunking

```rust
use ragcraft::Chunker;
use std::collections::HashMap;

fn main() {
    let chunker = Chunker::new(None);
    let markdown = "# Guide\nGetting started with Rust.\n## Installation\nInstall Rust via rustup.";
    
    let chunks = chunker.chunk_document(
        markdown,
        "doc-guide",
        50, // max tokens per chunk
        0,  // overlap
        "md",
        &HashMap::new(),
    );

    for chunk in chunks {
        println!("Chunk [{}..{}]: {}", chunk.start_line, chunk.end_line, chunk.content);
    }
}
```

---

### 3. Persistent Vector Store on Disk

```rust
use ragcraft::{StoredDocument, VectorStore};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store_path = Path::new(".cache/vectors.bin");

    // Load existing store or initialize empty
    let mut store = VectorStore::load_or_create(store_path)?;

    store.add(
        StoredDocument::new("arch-1", "src/main.rs", "fn main() { println!(\"Hello\"); }", "code"),
        vec![0.05, 0.12, 0.99],
    );

    // Atomically persist to disk
    store.save_to_disk(store_path)?;

    println!("Vector store has {} documents", store.len());
    Ok(())
}
```

---

## Sponsor & Support

Developed by **[Kreive Tech](https://kreive.com.br)**.

For questions, enterprise agent support, or contributions, reach out to `contact@kreive.com.br`.

---

## License

MIT © [Kreive Tech](https://kreive.com.br)
