
const NUM_PERM: usize = 128;
const SHINGLE_SIZE: usize = 3; // character n-grams

/// A MinHash signature for approximate Jaccard similarity.
#[derive(Clone, Debug)]
pub struct MinHashSignature {
    values: [u32; NUM_PERM],
}

/// Generate universal hash function coefficients: (a * x + b) % LARGE_PRIME
struct HashFunc {
    a: u32,
    b: u32,
}

impl HashFunc {
    fn hash(&self, x: u32) -> u32 {
        // Using a large Mersenne-like prime for modulo
        const PRIME: u64 = 4_294_967_291; // largest prime < 2^32
        ((self.a as u64 * x as u64 + self.b as u64) % PRIME) as u32
    }
}

/// Generate a MinHash signature from a set of shingles.
pub fn signature(text: &str) -> MinHashSignature {
    let shingles = shingle(text);
    let funcs = hash_funcs();
    let mut sig = [u32::MAX; NUM_PERM];

    for shingle in &shingles {
        let h = shingle_hash(shingle);
        for (i, func) in funcs.iter().enumerate() {
            let v = func.hash(h);
            if v < sig[i] {
                sig[i] = v;
            }
        }
    }

    // If no shingles, zero out
    if shingles.is_empty() {
        sig = [0; NUM_PERM];
    }

    MinHashSignature { values: sig }
}

/// Estimate Jaccard similarity between two signatures.
pub fn jaccard_estimate(a: &MinHashSignature, b: &MinHashSignature) -> f64 {
    let matches = a
        .values
        .iter()
        .zip(b.values.iter())
        .filter(|(x, y)| x == y)
        .count();
    matches as f64 / NUM_PERM as f64
}

fn shingle(text: &str) -> Vec<String> {
    let normalized: String = text
        .trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();

    if normalized.len() < SHINGLE_SIZE {
        return vec![normalized];
    }

    (0..=normalized.len().saturating_sub(SHINGLE_SIZE))
        .map(|i| normalized[i..i + SHINGLE_SIZE].to_string())
        .collect()
}

fn shingle_hash(s: &str) -> u32 {
    // Simple FNV-1a-like hash
    let mut h: u32 = 2166136261;
    for byte in s.bytes() {
        h ^= byte as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

fn hash_funcs() -> Vec<HashFunc> {
    // Deterministic coefficients via fixed seed
    let _seed_placeholder = 0u8; // deterministic hash funcs below
    let mut seed = [0u8; 32];
    // Use a deterministic seed for reproducibility
    for (i, byte) in seed.iter_mut().enumerate() {
        *byte = ((i as u64 * 1103515245 + 12345) & 0xFF) as u8;
    }
    // Generate deterministic hash functions
    (0..NUM_PERM)
        .map(|i| {
            let a = (i as u64).wrapping_mul(1103515245).wrapping_add(12345).rem_euclid(4_294_967_291) as u32;
            let b = ((i as u64).wrapping_add(1000)).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407).rem_euclid(4_294_967_291) as u32;
            HashFunc { a, b }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_high_similarity() {
        let a = signature("check disk usage on the server");
        let b = signature("check disk usage on the server");
        assert!(jaccard_estimate(&a, &b) > 0.99);
    }

    #[test]
    fn near_identical_text_high_similarity() {
        let a = signature("check disk usage on the server");
        let b = signature("check disk usage on server");
        let sim = jaccard_estimate(&a, &b);
        assert!(sim > 0.7, "expected > 0.7, got {}", sim);
    }

    #[test]
    fn different_text_low_similarity() {
        let a = signature("check disk usage");
        let b = signature("deploy kubernetes cluster");
        let sim = jaccard_estimate(&a, &b);
        assert!(sim < 0.4, "expected < 0.4, got {}", sim);
    }

    #[test]
    fn empty_text() {
        let a = signature("");
        let b = signature("hello");
        let sim = jaccard_estimate(&a, &b);
        // Empty vs non-empty should have low similarity
        assert!(sim < 0.5);
    }
}
