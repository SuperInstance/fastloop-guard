use fastloop_guard::cache::QueryCache;
use std::time::Duration;

#[test]
fn full_three_gate_flow() {
    let cache = QueryCache::new(100, Duration::from_secs(3600));

    // Miss
    let (gate, resp) = cache.lookup("check disk usage", 0.95);
    assert_eq!(gate, 0);
    assert!(resp.is_none());

    // Insert
    cache.insert("check disk usage", "df -h");

    // Exact hit (gate 1)
    let (gate, resp) = cache.lookup("check disk usage", 0.95);
    assert_eq!(gate, 1);
    assert_eq!(resp.unwrap(), "df -h");

    // Case-insensitive exact hit
    let (gate, resp) = cache.lookup("Check Disk Usage", 0.95);
    assert_eq!(gate, 1);
    assert_eq!(resp.unwrap(), "df -h");
}

#[test]
fn fuzzy_match_gate_2() {
    let cache = QueryCache::new(100, Duration::from_secs(3600));
    cache.insert("check disk usage on the server", "df -h");

    // Different wording but similar meaning — should fuzzy match at low threshold
    let (gate, _resp) = cache.lookup("check the disk usage on the server", 0.5);
    assert!(gate > 0, "expected a hit (gate 1 or 2), got gate {}", gate);
}

#[test]
fn stats_accuracy() {
    let cache = QueryCache::new(100, Duration::from_secs(3600));
    cache.insert("q1", "r1");

    cache.lookup("q1", 0.95); // exact hit
    cache.lookup("q2", 0.95); // miss

    let (hits, misses, rate) = cache.stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);
    assert!((rate - 0.5).abs() < 0.01);
}
