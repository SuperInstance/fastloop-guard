use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use crate::hash::fingerprint;
use crate::similarity::{jaccard_estimate, signature, MinHashSignature};

struct CacheEntry {
    query: String,
    response: String,
    sig: MinHashSignature,
    inserted: Instant,
}

pub struct QueryCache {
    /// Exact-match LRU cache keyed by BLAKE2b hash.
    exact: Mutex<LruCache<[u8; 32], CacheEntry>>,
    /// TTL for cache entries.
    ttl: Duration,
    /// Stats counters.
    hits: Mutex<u64>,
    misses: Mutex<u64>,
}

impl QueryCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            exact: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1024).unwrap()),
            )),
            ttl,
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        }
    }

    /// Three-gate lookup: exact → fuzzy → miss.
    /// Returns (gate, response) where gate 1=exact, 2=fuzzy, 0=miss.
    pub fn lookup(&self, query: &str, threshold: f64) -> (u8, Option<String>) {
        let fp = fingerprint(query);

        // Gate 1: Exact match
        {
            let mut cache = self.exact.lock();
            if let Some(entry) = cache.get(&fp) {
                if entry.inserted.elapsed() < self.ttl {
                    *self.hits.lock() += 1;
                    return (1, Some(entry.response.clone()));
                } else {
                    cache.pop(&fp);
                }
            }
        }

        // Gate 2: Fuzzy match via MinHash
        let sig = signature(query);
        {
            let cache = self.exact.lock();
            // Iterate recent entries to find a fuzzy match
            for (_, entry) in cache.iter() {
                if entry.inserted.elapsed() >= self.ttl {
                    continue;
                }
                let sim = jaccard_estimate(&sig, &entry.sig);
                if sim >= threshold {
                    *self.hits.lock() += 1;
                    return (2, Some(entry.response.clone()));
                }
            }
        }

        // Gate 0: Miss
        *self.misses.lock() += 1;
        (0, None)
    }

    /// Insert a query→response pair into the cache.
    pub fn insert(&self, query: &str, response: &str) {
        let fp = fingerprint(query);
        let sig = signature(query);
        let entry = CacheEntry {
            query: query.to_string(),
            response: response.to_string(),
            sig,
            inserted: Instant::now(),
        };
        let mut cache = self.exact.lock();
        cache.put(fp, entry);
    }

    /// Get stats snapshot.
    pub fn stats(&self) -> (u64, u64, f64) {
        let hits = *self.hits.lock();
        let misses = *self.misses.lock();
        let total = hits + misses;
        let rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        (hits, misses, rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_hit() {
        let cache = QueryCache::new(100, Duration::from_secs(3600));
        cache.insert("check disk usage", "df -h");
        let (gate, resp) = cache.lookup("check disk usage", 0.95);
        assert_eq!(gate, 1);
        assert_eq!(resp.unwrap(), "df -h");
    }

    #[test]
    fn miss_then_insert() {
        let cache = QueryCache::new(100, Duration::from_secs(3600));
        let (gate, resp) = cache.lookup("hello", 0.95);
        assert_eq!(gate, 0);
        assert!(resp.is_none());
        cache.insert("hello", "world");
        let (gate, resp) = cache.lookup("hello", 0.95);
        assert_eq!(gate, 1);
        assert_eq!(resp.unwrap(), "world");
    }

    #[test]
    fn expired_entry_is_miss() {
        let cache = QueryCache::new(100, Duration::from_millis(10));
        cache.insert("old query", "old response");
        std::thread::sleep(Duration::from_millis(20));
        let (gate, _) = cache.lookup("old query", 0.95);
        assert_eq!(gate, 0); // expired → miss
    }

    #[test]
    fn stats_track() {
        let cache = QueryCache::new(100, Duration::from_secs(3600));
        cache.insert("q1", "r1");
        cache.lookup("q1", 0.95); // hit
        cache.lookup("q2", 0.95); // miss
        let (hits, misses, rate) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert!((rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn lru_eviction() {
        let cache = QueryCache::new(2, Duration::from_secs(3600));
        cache.insert("a", "ra");
        cache.insert("b", "rb");
        cache.insert("c", "rc"); // should evict "a"
        let (gate, _) = cache.lookup("a", 0.95);
        assert_eq!(gate, 0); // evicted → miss
        let (gate, _) = cache.lookup("c", 0.95);
        assert_eq!(gate, 1); // still present
    }

    #[test]
    fn case_insensitive_exact() {
        let cache = QueryCache::new(100, Duration::from_secs(3600));
        cache.insert("Check Disk", "df -h");
        let (gate, resp) = cache.lookup("check disk", 0.95);
        assert_eq!(gate, 1);
        assert_eq!(resp.unwrap(), "df -h");
    }
}
