use serde::{Deserialize, Serialize};

/// Client → Daemon: cache lookup.
#[derive(Debug, Deserialize)]
pub struct CacheRequest {
    pub query: String,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
}

fn default_threshold() -> f64 {
    0.95
}

/// Client → Daemon: store a response in the cache.
#[derive(Debug, Deserialize)]
pub struct StoreRequest {
    pub query: String,
    pub response: String,
}

/// Client → Daemon: stats request.
#[derive(Debug, Deserialize)]
pub struct StatsRequest {
    pub stats: bool,
}

/// Daemon → Client: cache lookup response.
#[derive(Debug, Serialize)]
pub struct CacheResponse {
    pub hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    pub gate: u8,
    pub latency_us: u64,
}

/// Daemon → Client: store acknowledgment.
#[derive(Debug, Serialize)]
pub struct StoreResponse {
    pub stored: bool,
}

/// Daemon → Client: stats snapshot.
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

/// Daemon → Client: error.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Top-level response frame.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Response {
    Cache(CacheResponse),
    Store(StoreResponse),
    Stats(StatsResponse),
    Error(ErrorResponse),
}

/// Top-level request frame.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum Request {
    #[serde(rename = "lookup")]
    Lookup(CacheRequest),
    #[serde(rename = "store")]
    Store(StoreRequest),
    #[serde(rename = "stats")]
    Stats(StatsRequest),
}
