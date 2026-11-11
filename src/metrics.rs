use crate::types::RetrievalResult;

/// Assembles a formatted context string for LLM prompts from retrieval results with default threshold (0.40).
pub fn assemble_context(results: &RetrievalResult) -> String {
    assemble_context_with_threshold(results, 0.40)
}

/// Assembles a formatted context string for LLM prompts filtering by a minimum relevance score threshold.
pub fn assemble_context_with_threshold(results: &RetrievalResult, min_threshold: f32) -> String {
    let mut ctx = String::from("## RELEVANT CONTEXT\n\n");
    let mut added = 0;
    for (doc, score) in results.documents.iter().zip(&results.scores) {
        if *score >= min_threshold {
            ctx.push_str(&format!("**Relevance: {:.2}%**\n\n", score * 100.0));
            ctx.push_str(doc);
            ctx.push_str("\n\n---\n\n");
            added += 1;
        }
    }
    if added == 0 {
        ctx.push_str("(No relevant documents found)\n");
    }
    ctx
}

/// Dot product of two vectors. If both are L2-normalized, this equals cosine similarity.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculates the L2 (Euclidean) norm of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalizes a vector in-place to unit length (L2 norm = 1.0).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Computes the cosine similarity between two arbitrary vectors.
///
/// Handles normalization automatically.
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a <= 1e-12 || norm_b <= 1e-12 {
        return 0.0;
    }
    dot_product(a, b) / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(len: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        v[idx] = 1.0;
        v
    }

    #[test]
    fn cosine_sim_identical_unit_vectors_is_one() {
        let v = unit(4, 0);
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal_vectors_is_zero() {
        let a = unit(4, 0);
        let b = unit(4, 1);
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    #[test]
    fn cosine_sim_zero_vector_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let z = vec![0.0f32, 0.0, 0.0];
        assert_eq!(cosine_sim(&a, &z), 0.0);
    }

    #[test]
    fn cosine_sim_mismatched_lengths_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0];
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    #[test]
    fn assemble_context_empty_results() {
        let result = RetrievalResult { documents: vec![], scores: vec![] };
        let ctx = assemble_context(&result);
        assert!(ctx.contains("No relevant documents found"));
    }

    #[test]
    fn assemble_context_formats_relevance_pct() {
        let result = RetrievalResult {
            documents: vec!["doc content".to_string()],
            scores: vec![0.75],
        };
        let ctx = assemble_context(&result);
        assert!(ctx.contains("75.00%"));
        assert!(ctx.contains("doc content"));
    }

    #[test]
    fn assemble_context_includes_all_documents() {
        let result = RetrievalResult {
            documents: vec!["alpha".into(), "beta".into(), "gamma".into()],
            scores:    vec![0.9, 0.7, 0.5],
        };
        let ctx = assemble_context(&result);
        assert!(ctx.contains("alpha"));
        assert!(ctx.contains("beta"));
        assert!(ctx.contains("gamma"));
        assert!(ctx.contains("90.00%"));
        assert!(ctx.contains("50.00%"));
    }

    #[test]
    fn assemble_context_has_separator_between_documents() {
        let result = RetrievalResult {
            documents: vec!["first".into(), "second".into()],
            scores:    vec![1.0, 0.8],
        };
        let ctx = assemble_context(&result);
        assert!(ctx.contains("---"));
    }

    #[test]
    fn assemble_context_starts_with_header() {
        let result = RetrievalResult { documents: vec![], scores: vec![] };
        assert!(assemble_context(&result).starts_with("## RELEVANT CONTEXT"));
    }

    #[test]
    fn cosine_sim_partial_overlap() {
        let a = [1.0f32, 1.0, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert!((sim - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5, "sim={sim}");
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
    }
}
