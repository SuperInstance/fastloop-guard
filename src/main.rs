mod engine;
mod failure_tracker;
mod rate_limiter;
mod validator;

use engine::{GuardEngine, InterceptRequest, InterceptResponse};
use tokio::net::UnixListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{info, error, warn};

const SOCK_PATH: &str = "/tmp/fastloop_guard.sock";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("fastloop_guard=info")
        .init();

    let _ = std::fs::remove_file(SOCK_PATH);

    let listener = match UnixListener::bind(SOCK_PATH) {
        Ok(l) => l,
        Err(e) => {
            error!("failed to bind {}: {}", SOCK_PATH, e);
            std::process::exit(1);
        }
    };

    info!("fastloop-guard listening on {}", SOCK_PATH);

    let engine = std::sync::Arc::new(GuardEngine::new());

    loop {
        match listener.accept().await {
            Ok((mut stream, _addr)) => {
                let engine = engine.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    match stream.read(&mut buf).await {
                        Ok(n) if n > 0 => {
                            let resp = match serde_json::from_slice::<InterceptRequest>(&buf[..n]) {
                                Ok(req) => engine.process(req),
                                Err(e) => InterceptResponse {
                                    action: "ROUTE_TO_DEEP_LOOP".into(),
                                    reason: format!("invalid request: {}", e),
                                    state_hash: None,
                                    duration_us: 0,
                                },
                            };
                            if let Ok(json) = serde_json::to_vec(&resp) {
                                let _ = stream.write_all(&json).await;
                            }
                        }
                        Ok(_) => {}
                        Err(e) => warn!("read error: {}", e),
                    }
                });
            }
            Err(e) => error!("accept error: {}", e),
        }
    }
}
