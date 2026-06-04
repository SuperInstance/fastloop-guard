mod cache;
mod gate;
mod hash;
mod protocol;
mod similarity;

use gate::Gate;
use protocol::{Request, Response};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tracing::{error, info, warn};

const SOCK_PATH: &str = "/tmp/fastloop.sock";

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

    let gate = std::sync::Arc::new(Gate::new());

    // Shutdown signal
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    tokio::spawn(shutdown_watcher(shutdown_tx));

    loop {
        tokio::select! {
            accept = listener.accept() => {
                match accept {
                    Ok((mut stream, _addr)) => {
                        let gate = gate.clone();
                        tokio::spawn(async move {
                            handle_connection(&mut stream, &gate).await;
                        });
                    }
                    Err(e) => error!("accept error: {}", e),
                }
            }
            _ = shutdown_rx.recv() => {
                info!("shutting down gracefully");
                break;
            }
        }
    }

    let _ = std::fs::remove_file(SOCK_PATH);
    info!("fastloop-guard stopped");
}

async fn handle_connection(
    stream: &mut tokio::net::UnixStream,
    gate: &Gate,
) {
    let mut buf = vec![0u8; 65536];
    match stream.read(&mut buf).await {
        Ok(n) if n > 0 => {
            let resp = match serde_json::from_slice::<Request>(&buf[..n]) {
                Ok(Request::Lookup(req)) => {
                    let resp = gate.lookup(&req.query, req.threshold);
                    Response::Cache(resp)
                }
                Ok(Request::Store(req)) => {
                    gate.insert(&req.query, &req.response);
                    Response::Store(protocol::StoreResponse { stored: true })
                }
                Ok(Request::Stats(_)) => gate.stats(),
                Err(e) => {
                    warn!("invalid request: {}", e);
                    Response::Error(protocol::ErrorResponse {
                        error: format!("invalid request: {}", e),
                    })
                }
            };

            if let Ok(json) = serde_json::to_vec(&resp) {
                let _ = stream.write_all(&json).await;
            }
        }
        Ok(_) => {}
        Err(e) => warn!("read error: {}", e),
    }
}

#[cfg(unix)]
async fn shutdown_watcher(tx: tokio::sync::mpsc::Sender<()>) {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    term.recv().await;
    let _ = tx.send(()).await;
}

#[cfg(not(unix))]
async fn shutdown_watcher(tx: tokio::sync::mpsc::Sender<()>) {
    tokio::signal::ctrl_c().await.ok();
    let _ = tx.send(()).await;
}
