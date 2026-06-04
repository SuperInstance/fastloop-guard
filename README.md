# fastloop-guard

A Unix Domain Socket daemon that sits between the user and LLM calls. It intercepts repeated identical or near-identical queries and returns cached responses instantly.

## Architecture

```
Client → fastloop-guard (UDS) → { hit: true/false, response, gate, latency_us }
```

### Three-Gate Lookup

| Gate | Method | Latency Target |
|------|--------|---------------|
| 1 — Exact | BLAKE2b hash → O(1) LRU lookup | < 50µs |
| 2 — Fuzzy | MinHash signature → Jaccard similarity ≥ threshold | < 200µs |
| 0 — Miss | Cache miss → return to caller | N/A |

## Protocol (JSON over UDS at `/tmp/fastloop.sock`)

### Lookup
```json
→ {"type":"lookup","query":"check disk usage","threshold":0.95}
← {"hit":true,"response":"df -h","gate":1,"latency_us":12}
```

### Store
```json
→ {"type":"store","query":"check disk usage","response":"df -h"}
← {"stored":true}
```

### Stats
```json
→ {"type":"stats","stats":true}
← {"hits":42,"misses":58,"hit_rate":0.42}
```

## Features

- **LRU cache** with configurable TTL (default: 1 hour, 4096 entries)
- **Exact match**: BLAKE2b-256 hash with case/whitespace normalization
- **Fuzzy match**: MinHash (128 permutations, 3-gram shingles) for Jaccard similarity
- **Stats endpoint**: hit/miss counts and hit rate
- **Graceful shutdown** on SIGTERM
- **Async I/O** with tokio

## Build & Run

```bash
cargo build --release
./target/release/fastloop-guard
```

## Test

```bash
cargo test                # unit + integration
cargo test -- --ignored   # integration (starts daemon)
```

## License

MIT
