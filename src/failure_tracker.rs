use parking_lot::Mutex;
use std::collections::HashMap;

pub struct FailureTracker {
    /// Consecutive failure counts per sandbox.
    counters: Mutex<HashMap<String, u32>>,
    /// Hash of failing state → error message (dedup / fast-reject).
    failure_db: Mutex<HashMap<String, String>>,
    max_failures: u32,
}

impl FailureTracker {
    pub fn new(max_failures: u32) -> Self {
        Self {
            counters: Mutex::new(HashMap::new()),
            failure_db: Mutex::new(HashMap::new()),
            max_failures,
        }
    }

    /// Record a failure. Returns `true` if the sandbox should be terminated.
    pub fn record_failure(&self, sandbox_id: &str, state_hash: &str, error: &str) -> bool {
        {
            let mut db = self.failure_db.lock();
            db.insert(state_hash.to_string(), error.to_string());
        }
        let mut counters = self.counters.lock();
        let count = counters.entry(sandbox_id.to_string()).or_insert(0);
        *count += 1;
        *count >= self.max_failures
    }

    /// Record a success — resets the consecutive failure counter.
    pub fn record_success(&self, sandbox_id: &str) {
        let mut counters = self.counters.lock();
        counters.remove(sandbox_id);
    }

    /// Check if a state hash is a known failure.
    pub fn is_known_failure(&self, state_hash: &str) -> Option<String> {
        let db = self.failure_db.lock();
        db.get(state_hash).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_failures() {
        let tracker = FailureTracker::new(3);
        assert!(!tracker.record_failure("s1", "hash1", "err"));
        assert!(!tracker.record_failure("s1", "hash2", "err"));
        assert!(tracker.record_failure("s1", "hash3", "err")); // 3rd triggers termination
    }

    #[test]
    fn success_resets() {
        let tracker = FailureTracker::new(3);
        tracker.record_failure("s1", "h1", "err");
        tracker.record_failure("s1", "h2", "err");
        tracker.record_success("s1");
        // Counter reset — takes 3 more to trigger
        assert!(!tracker.record_failure("s1", "h3", "err"));
        assert!(!tracker.record_failure("s1", "h4", "err"));
        assert!(tracker.record_failure("s1", "h5", "err"));
    }

    #[test]
    fn known_failure_lookup() {
        let tracker = FailureTracker::new(3);
        tracker.record_failure("s1", "abc123", "parse error");
        assert_eq!(tracker.is_known_failure("abc123"), Some("parse error".into()));
        assert_eq!(tracker.is_known_failure("unknown"), None);
    }

    #[test]
    fn independent_sandboxes() {
        let tracker = FailureTracker::new(2);
        assert!(!tracker.record_failure("a", "h1", "e"));
        assert!(!tracker.record_failure("b", "h2", "e"));
        assert!(tracker.record_failure("a", "h3", "e"));  // a hits limit of 2
        assert!(tracker.record_failure("b", "h4", "e"));  // b also hits limit of 2
    }
}
