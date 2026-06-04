use crate::cache::QueryCache;
use crate::protocol::{CacheResponse, Response, StatsResponse};
use std::time::Duration;

/// Default cache capacity.
const DEFAULT_CAPACITY: usize = 4096;
/// Default TTL: 1 hour.
const DEFAULT_TTL: Duration = Duration::from_secs(3600);

/// The guard gatekeeper: owns the cache and handles requests.
pub struct Gate {
    cache: QueryCache,
}

impl Gate {
    pub fn new() -> Self {
        Self {
            cache: QueryCache::new(DEFAULT_CAPACITY, DEFAULT_TTL),
        }
    }

    pub fn with_params(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: QueryCache::new(capacity, ttl),
        }
    }

    /// Process a cache lookup request through the three gates.
    /// Returns the response; caller should measure latency externally.
    pub fn lookup(&self, query: &str, threshold: f64) -> CacheResponse {
        let start = std::time::Instant::now();
        let (gate, response) = self.cache.lookup(query, threshold);
        let latency_us = start.elapsed().as_micros() as u64;

        CacheResponse {
            hit: gate > 0,
            response,
            gate,
            latency_us,
        }
    }

    /// Insert a new entry into the cache (called on miss, after getting the real response).
    pub fn insert(&self, query: &str, response: &str) {
        self.cache.insert(query, response);
    }

    /// Get stats.
    pub fn stats(&self) -> Response {
        let (hits, misses, hit_rate) = self.cache.stats();
        Response::Stats(StatsResponse {
            hits,
            misses,
            hit_rate,
        })
    }
}
