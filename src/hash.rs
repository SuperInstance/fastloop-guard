use blake2::{Blake2b512, Digest};

/// Compute a BLAKE2b-512 fingerprint of a normalized query string, truncated to 32 bytes.
pub fn fingerprint(query: &str) -> [u8; 32] {
    let normalized = normalize(query);
    let mut hasher = Blake2b512::new();
    hasher.update(normalized.as_bytes());
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result[..32]);
    out
}

/// Normalize whitespace and case for consistent hashing.
fn normalize(query: &str) -> String {
    query
        .trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_query_same_hash() {
        let a = fingerprint("check disk usage");
        let b = fingerprint("check disk usage");
        assert_eq!(a, b);
    }

    #[test]
    fn whitespace_normalization() {
        let a = fingerprint("check   disk\tusage");
        let b = fingerprint("check disk usage");
        assert_eq!(a, b);
    }

    #[test]
    fn case_insensitive() {
        let a = fingerprint("Check Disk Usage");
        let b = fingerprint("check disk usage");
        assert_eq!(a, b);
    }

    #[test]
    fn different_queries_differ() {
        let a = fingerprint("check disk usage");
        let b = fingerprint("check memory usage");
        assert_ne!(a, b);
    }
}
