# Future Integration: fastloop-guard

## Current State
A Unix Domain Socket daemon that intercepts repeated identical/near-identical LLM queries and returns cached responses instantly. Three-gate lookup: Gate 1 exact BLAKE2b hash (<50µs), Gate 2 fuzzy MinHash/Jaccard (<200µs), Gate 0 miss (pass through).

## Integration Opportunities

### With room event loop protection
fastloop-guard protects room event loops from degenerate behavior. If a room's cells keep asking the same question (e.g., "what's my neighbor's state?"), the guard intercepts and returns cached answers. The room's event loop stays healthy even when cells are in a loop.

### With llm-proxy caching
The guard sits in front of the llm-proxy for LLM query caching. If multiple rooms ask similar questions, the guard returns cached answers without hitting the LLM. This reduces LLM costs and latency dramatically for common queries.

### With lever-runner
lever-runner's three-gate pipeline and fastloop-guard's three-gate cache are complementary. lever-runner matches commands; fastloop-guard caches LLM queries. Together they form a complete request pipeline: command matching → execution → LLM query (if needed) → cache check → response.

## Dormant Ideas Now Unlockable
The guard was for LLM query caching. Now it protects room event loops too. A room with fastloop-guard can run indefinitely without degenerating — repeated queries are caught and cached.

## Potential in Mature Systems
fastloop-guard is deployed alongside every room runtime. It monitors the room's query patterns, caches repeated queries, and alerts when a room is in a degenerate loop. The guard IS the room's immune system against infinite loops.

## Cross-Pollination Ideas
- **llm-proxy**: Guard caches proxy queries
- **lever-runner**: Complementary three-gate pipelines
- **captains-log**: Guard alerts logged as fleet events

## Dependencies for Next Steps
- Integration with room event loop
- Pattern detection for degenerate room behavior
- Fleet-wide cache sharing between rooms
