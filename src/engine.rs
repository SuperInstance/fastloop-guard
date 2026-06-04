use crate::failure_tracker::FailureTracker;
use crate::rate_limiter::SlidingWindowLimiter;
use crate::validator;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;

#[derive(Debug, Deserialize)]
pub struct InterceptRequest {
    pub r#type: String,
    pub payload: String,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub sandbox_id: String,
}

#[derive(Debug, Serialize)]
pub struct InterceptResponse {
    pub action: String,
    pub reason: String,
    pub state_hash: Option<String>,
    pub duration_us: u64,
}

fn hash_state(payload: &str, context: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    hasher.update(context.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub struct GuardEngine {
    limiter: SlidingWindowLimiter,
    tracker: FailureTracker,
}

impl GuardEngine {
    pub fn new() -> Self {
        Self {
            limiter: SlidingWindowLimiter::new(100, Duration::from_secs(60)),
            tracker: FailureTracker::new(3),
        }
    }

    #[allow(dead_code)]
    pub fn with_limits(max_requests: usize, window: Duration, max_failures: u32) -> Self {
        Self {
            limiter: SlidingWindowLimiter::new(max_requests, window),
            tracker: FailureTracker::new(max_failures),
        }
    }

    pub fn process(&self, req: InterceptRequest) -> InterceptResponse {
        use std::time::Instant;
        let start = Instant::now();

        // 1. Rate limit check
        if self.limiter.check(&req.sandbox_id) {
            return InterceptResponse {
                action: "ROUTE_TO_DEEP_LOOP".into(),
                reason: "rate limited".into(),
                state_hash: None,
                duration_us: start.elapsed().as_micros() as u64,
            };
        }

        // 2. Hash the state
        let state_hash = hash_state(&req.payload, &req.context);

        // 3. Check failure_db for cached failures
        if let Some(err) = self.tracker.is_known_failure(&state_hash) {
            return InterceptResponse {
                action: "ROUTE_TO_DEEP_LOOP".into(),
                reason: format!("known failure: {}", err),
                state_hash: Some(state_hash),
                duration_us: start.elapsed().as_micros() as u64,
            };
        }

        // 4. Validate based on type
        let validation_result = match req.r#type.as_str() {
            "python" => validator::validate_python(&req.payload),
            "json" => validator::validate_json(&req.payload),
            "rust" => validator::validate_rust(&req.payload),
            other => Err(format!("unknown type: {}", other)),
        };

        // 5. Update failure tracker & return decision
        match validation_result {
            Ok(()) => {
                self.tracker.record_success(&req.sandbox_id);
                InterceptResponse {
                    action: "EXECUTE_IMMEDIATELY".into(),
                    reason: "validation passed".into(),
                    state_hash: Some(state_hash),
                    duration_us: start.elapsed().as_micros() as u64,
                }
            }
            Err(err) => {
                let terminated = self.tracker.record_failure(
                    &req.sandbox_id,
                    &state_hash,
                    &err,
                );
                InterceptResponse {
                    action: if terminated {
                        "SANDBOX_TERMINATED".into()
                    } else {
                        "ROUTE_TO_DEEP_LOOP".into()
                    },
                    reason: format!("validation failed: {}", err),
                    state_hash: Some(state_hash),
                    duration_us: start.elapsed().as_micros() as u64,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_python_executes() {
        let engine = GuardEngine::with_limits(100, Duration::from_secs(60), 3);
        let resp = engine.process(InterceptRequest {
            r#type: "python".into(),
            payload: "x = 1 + 2".into(),
            context: "".into(),
            sandbox_id: "test".into(),
        });
        assert_eq!(resp.action, "EXECUTE_IMMEDIATELY");
    }

    #[test]
    fn invalid_python_routes_to_deep_loop() {
        let engine = GuardEngine::with_limits(100, Duration::from_secs(60), 3);
        let resp = engine.process(InterceptRequest {
            r#type: "python".into(),
            payload: "os.system('rm -rf /')".into(),
            context: "".into(),
            sandbox_id: "test".into(),
        });
        assert_eq!(resp.action, "ROUTE_TO_DEEP_LOOP");
        assert!(resp.reason.contains("blocked pattern"));
    }

    #[test]
    fn cascading_failures_terminate() {
        let engine = GuardEngine::with_limits(100, Duration::from_secs(60), 3);
        // Use different payloads so known_failure cache doesn't short-circuit
        for i in 0..2 {
            let resp = engine.process(InterceptRequest {
                r#type: "python".into(),
                payload: format!("os.system('x{}')", i),
                context: "".into(),
                sandbox_id: "s1".into(),
            });
            assert_eq!(resp.action, "ROUTE_TO_DEEP_LOOP");
        }
        let resp = engine.process(InterceptRequest {
            r#type: "python".into(),
            payload: "os.system('x_final')".into(),
            context: "".into(),
            sandbox_id: "s1".into(),
        });
        assert_eq!(resp.action, "SANDBOX_TERMINATED");
    }

    #[test]
    fn rate_limited() {
        let engine = GuardEngine::with_limits(2, Duration::from_secs(60), 3);
        for _ in 0..2 {
            let resp = engine.process(InterceptRequest {
                r#type: "json".into(),
                payload: "{}".into(),
                context: "".into(),
                sandbox_id: "limited".into(),
            });
            assert_eq!(resp.action, "EXECUTE_IMMEDIATELY");
        }
        let resp = engine.process(InterceptRequest {
            r#type: "json".into(),
            payload: "{}".into(),
            context: "".into(),
            sandbox_id: "limited".into(),
        });
        assert_eq!(resp.action, "ROUTE_TO_DEEP_LOOP");
        assert!(resp.reason.contains("rate limited"));
    }

    #[test]
    fn known_failure_cached() {
        let engine = GuardEngine::with_limits(100, Duration::from_secs(60), 3);
        engine.process(InterceptRequest {
            r#type: "python".into(),
            payload: "os.system('x')".into(),
            context: "ctx".into(),
            sandbox_id: "s1".into(),
        });
        let resp = engine.process(InterceptRequest {
            r#type: "python".into(),
            payload: "os.system('x')".into(),
            context: "ctx".into(),
            sandbox_id: "s2".into(),
        });
        assert!(resp.reason.contains("known failure"));
    }

    #[test]
    fn unknown_type_rejected() {
        let engine = GuardEngine::new();
        let resp = engine.process(InterceptRequest {
            r#type: "brainfuck".into(),
            payload: "+++".into(),
            context: "".into(),
            sandbox_id: "test".into(),
        });
        assert_eq!(resp.action, "ROUTE_TO_DEEP_LOOP");
        assert!(resp.reason.contains("unknown type"));
    }

    #[test]
    fn success_resets_failures() {
        let engine = GuardEngine::with_limits(100, Duration::from_secs(60), 3);
        // 2 failures with different payloads
        engine.process(InterceptRequest {
            r#type: "python".into(),
            payload: "os.system('a')".into(),
            context: "".into(),
            sandbox_id: "s1".into(),
        });
        engine.process(InterceptRequest {
            r#type: "python".into(),
            payload: "os.system('b')".into(),
            context: "".into(),
            sandbox_id: "s1".into(),
        });
        // Success resets counter
        engine.process(InterceptRequest {
            r#type: "json".into(),
            payload: "{}".into(),
            context: "".into(),
            sandbox_id: "s1".into(),
        });
        // 3 more failures with new payloads
        for i in 0..2 {
            let resp = engine.process(InterceptRequest {
                r#type: "python".into(),
                payload: format!("os.system('c{}')", i),
                context: "".into(),
                sandbox_id: "s1".into(),
            });
            assert_eq!(resp.action, "ROUTE_TO_DEEP_LOOP", "failure {}", i);
        }
        let resp = engine.process(InterceptRequest {
            r#type: "python".into(),
            payload: "os.system('c_final')".into(),
            context: "".into(),
            sandbox_id: "s1".into(),
        });
        assert_eq!(resp.action, "SANDBOX_TERMINATED");
    }
}
