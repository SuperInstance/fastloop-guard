use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct SlidingWindowLimiter {
    windows: Mutex<HashMap<String, Vec<Instant>>>,
    max_requests: usize,
    window_duration: Duration,
}

impl SlidingWindowLimiter {
    pub fn new(max_requests: usize, window_duration: Duration) -> Self {
        Self {
            windows: Mutex::new(HashMap::new()),
            max_requests,
            window_duration,
        }
    }

    /// Returns `true` if the request is rate-limited (over limit).
    pub fn check(&self, id: &str) -> bool {
        let now = Instant::now();
        let cutoff = now - self.window_duration;
        let mut windows = self.windows.lock();

        let entry = windows.entry(id.to_string()).or_default();
        // Remove timestamps older than the window
        entry.retain(|t| *t > cutoff);

        if entry.len() >= self.max_requests {
            return true; // rate limited
        }

        entry.push(now);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_up_to_limit() {
        let limiter = SlidingWindowLimiter::new(3, Duration::from_secs(60));
        assert!(!limiter.check("sandbox-1"));
        assert!(!limiter.check("sandbox-1"));
        assert!(!limiter.check("sandbox-1"));
        assert!(limiter.check("sandbox-1")); // 4th is blocked
    }

    #[test]
    fn different_sandboxes_independent() {
        let limiter = SlidingWindowLimiter::new(2, Duration::from_secs(60));
        assert!(!limiter.check("a"));
        assert!(!limiter.check("a"));
        assert!(!limiter.check("b")); // different sandbox, allowed
        assert!(limiter.check("a"));  // a is over limit
    }

    #[test]
    fn window_expiry() {
        let limiter = SlidingWindowLimiter::new(1, Duration::from_millis(50));
        assert!(!limiter.check("x"));
        assert!(limiter.check("x"));
        std::thread::sleep(Duration::from_millis(60));
        assert!(!limiter.check("x")); // window expired, allowed again
    }
}
