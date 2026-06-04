use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;

const SOCK_PATH: &str = "/tmp/fastloop.sock";

struct DaemonGuard(Child);

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
        let _ = std::fs::remove_file(SOCK_PATH);
    }
}

fn start_daemon() -> DaemonGuard {
    // Build first
    let status = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("cargo build failed");
    assert!(status.success());

    let child = Command::new("./target/release/fastloop-guard")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .spawn()
        .expect("failed to start daemon");

    // Wait for socket to appear
    for _ in 0..50 {
        if std::path::Path::new(SOCK_PATH).exists() {
            return DaemonGuard(child);
        }
        thread::sleep(Duration::from_millis(100));
    }

    panic!("daemon did not start (socket not found)");
}

fn send_request(payload: &str) -> String {
    let mut stream = UnixStream::connect(SOCK_PATH).expect("connect failed");
    stream
        .write_all(payload.as_bytes())
        .expect("write failed");
    stream.shutdown(std::net::Shutdown::Write).ok();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).expect("read failed");
    resp
}

#[test]
fn test_integration_full_flow() {
    let _daemon = start_daemon();

    // 1. Store a response
    let store_req = r#"{"type":"store","query":"check disk usage","response":"df -h"}"#;
    let resp = send_request(store_req);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert!(v["stored"].as_bool().unwrap());

    // 2. Exact hit
    let lookup_req = r#"{"type":"lookup","query":"check disk usage","threshold":0.95}"#;
    let resp = send_request(lookup_req);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert!(v["hit"].as_bool().unwrap());
    assert_eq!(v["gate"].as_u64().unwrap(), 1);
    assert_eq!(v["response"].as_str().unwrap(), "df -h");
    assert!(v["latency_us"].as_u64().unwrap() < 50_000); // < 50ms

    // 3. Miss
    let miss_req = r#"{"type":"lookup","query":"something completely different","threshold":0.95}"#;
    let resp = send_request(miss_req);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert!(!v["hit"].as_bool().unwrap());
    assert_eq!(v["gate"].as_u64().unwrap(), 0);

    // 4. Stats
    let stats_req = r#"{"type":"stats","stats":true}"#;
    let resp = send_request(stats_req);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert!(v["hits"].as_u64().unwrap() >= 1);
    assert!(v["misses"].as_u64().unwrap() >= 1);
}
