use fastloop_guard::similarity::{jaccard_estimate, signature};

#[test]
fn identical_perfect_match() {
    let a = signature("check disk usage");
    let b = signature("check disk usage");
    assert!(jaccard_estimate(&a, &b) > 0.99);
}

#[test]
fn near_duplicate_high_similarity() {
    let a = signature("list all running docker containers");
    let b = signature("list all running containers docker");
    let sim = jaccard_estimate(&a, &b);
    assert!(sim > 0.5, "expected > 0.5, got {}", sim);
}

#[test]
fn unrelated_low_similarity() {
    let a = signature("check disk usage");
    let b = signature("deploy kubernetes cluster to production");
    let sim = jaccard_estimate(&a, &b);
    assert!(sim < 0.4, "expected < 0.4, got {}", sim);
}

#[test]
fn empty_strings() {
    let a = signature("");
    let b = signature("");
    let sim = jaccard_estimate(&a, &b);
    assert!(sim > 0.99); // both empty → identical
}

#[test]
fn single_char() {
    let a = signature("a");
    let b = signature("a");
    assert!(jaccard_estimate(&a, &b) > 0.99);
}
