# The Verification Gap: Proving GPU Code Correct

> Claude Code Opus essay on verifying agent-generated GPU programs.
> fastloop-guard is the last line of defense before GPU dispatch.

---

# The Verification Gap: Trusting Agent-Generated GPU Kernels

> *A research essay on the hardest unsolved problem in the Flux→PTX pipeline.*
> *When no human wrote the code and no human can read the output, what does correctness even mean?*

---

## I. The Problem, Stated Precisely

There is a moment in the Flux→PTX pipeline that ought to feel uncomfortable to everyone who has thought about it. An agent — a language model, a planning system, a reinforcement-learning controller — generates Flux bytecode expressing some computational intent. That bytecode passes through a compiler stack: Flux→MIR→Pliron→NVVM→LLVM→PTX. Somewhere in that chain, the agent's intent becomes silicon instructions that execute inside a GPU streaming multiprocessor at femtojoule per operation. No human wrote the Flux bytecode. No human can read the PTX. The agent might have generated wrong intent. The compiler might have silently introduced a semantic shift. The resulting kernel runs at 400,000 operations per second and the outputs flow into downstream systems.

How do you know it's right?

This is the verification gap. It is not the same problem as compiler correctness — that is a solvable, largely solved problem in the domain of certified compilers like CompCert. It is not the same problem as GPU debugging — CUDA-memcheck and compute sanitizers exist. The verification gap is specifically the problem of establishing *semantic correspondence* between what an agent intended and what the compiled kernel computes, in a world where neither the intent nor the compiled output can be audited by a human in the loop.

Three independent AI analysis sessions — DeepSeek V4 Flash, ByteDance Seed 2.0 Mini, and NousResearch Hermes 3 — each independently arrived at the same conclusion when analyzing the cuda-oxide/cudaclaw architecture: the single most dangerous unsolved problem is not warp divergence, not CRDT convergence overhead, not construct loading latency. It is verification. DeepSeek put it directly: "If agents generate their own GPU code, who verifies it?" Hermes noted that conservation laws at GPU scale "could inform the development of algorithms and hardware that naturally conserve quantities." The Synthesis section of the architectural thinking document names it as Gap #1 in the design.

This essay is an attempt to characterize the gap precisely and then to catalog what tools already exist — or can be built — to close it.

---

## II. Why Standard Verification Fails Here

The standard toolkit for verifying GPU kernels assumes a human at some point in the loop. Code review assumes the source is legible. Unit tests assume someone chose representative test cases. Property-based testing assumes someone specified the properties. Formal verification assumes someone wrote the specification. Every existing verification methodology has a human at the root of the trust chain.

Agent-generated code breaks this assumption at a structural level. The agent's "intent" is an implicit distribution over likely correct behaviors, not an explicit specification. When a language model generates Flux bytecode for a ternary attention mechanism, it is sampling from its posterior over what such a computation ought to look like — it is not executing a deterministic algorithm against a formal spec. The generated code is *plausibly correct*, not *provably correct*. And plausible correctness is exactly what testing is supposed to catch, not what it assumes.

The second failure is the PTX opacity problem. Parallel Thread Execution (PTX) is an intermediate assembly language for NVIDIA GPUs. It is nominally human-readable — it has labeled registers, typed instructions, explicit memory address spaces. But a compiled ternary attention kernel runs to tens of thousands of lines of PTX, with register allocation, loop unrolling, warp-level shuffle intrinsics, and predicate logic that no human can trace back to the original intent without tooling that does not yet exist. The final compiled SASS (Streaming Assembler) is entirely opaque without reverse-engineering tools. When the cudaclaw runtime loads this kernel and runs it persistently across 10,000 agents, any semantic error is silent.

The third failure is the intent/implementation gap, which is arguably the hardest. In human-written code, when a kernel produces wrong results, you can diff the source against the specification. With agent-generated code, the specification *is* the agent's internal state at generation time — a vector of activations in a transformer that no post-hoc analysis can recover. The agent might have generated precisely what it intended, and the intent itself might be wrong. This is not a bug in the compiler or the runtime. It is a semantic error that no static analysis can detect because no ground truth exists to compare against.

This distinction — between *implementation errors* (the code does not match the intent) and *intent errors* (the intent itself is wrong) — is the crux of the verification gap. Formal methods, statistical testing, and runtime invariants can address implementation errors. Intent errors require something deeper: an independent oracle that can evaluate whether the agent's expressed computation is the right computation for the task at hand.

---

## III. The Ternary Type System as a Verification Instrument

The most immediately tractable piece of the verification gap is the one that the ternary ecosystem unwittingly solved: the state-space reduction problem.

Consider what it means to exhaustively test a GPU kernel. For a function over 32-bit floats, the input domain for even a single element is 2^32 ≈ 4.3 billion values. For a vector of K elements, exhaustive testing is categorically impossible. This is why GPU testing has always relied on sampling — a few thousand representative inputs, some edge cases (NaN, ±inf, denormals), and the engineering judgment that the sampled behavior generalizes.

For a function over ternary values {-1, 0, +1}, the input domain for a single element is exactly 3 values. For a vector of K elements, the input domain is 3^K. This is still exponential, but the base is much smaller. For K=8 (a single byte of packed trits), the entire domain has 3^8 = 6,561 elements. For K=16, it is 43,046,721 — still tractable on modern hardware. For K=32 (a warp-width ternary vector), exhaustive testing produces 1.85 × 10^15 cases, which is not tractable on current hardware but is a fixed, bounded number rather than a conceptually infinite one.

More importantly, the ternary type system enables *property-based exhaustive verification* for small kernels. Rather than asking "does this kernel produce the right output on these test inputs," we can ask "does this kernel satisfy the following algebraic properties for all inputs in the domain?" For K ≤ 8, we can verify this by exhaustive enumeration. The `ternary-core` crate already provides the algebraic properties: commutativity and associativity of `tadd`, distributivity of `tmul` over `tadd`, the absorbing element (zero), the multiplicative identity (+1). A verification harness can generate all 3^8 input pairs and check every algebraic law in under a millisecond.

This is not mere unit testing. It is **complete verification** for the algebraic properties of small ternary kernels. No sampling, no statistical inference — every case checked.

The `ternary-tensor` crate extends this to multi-dimensional arrays. Its `matmul` function implements the triple loop with integer accumulation and ternary clamping. For matrices up to 3×3, exhaustive verification of the matmul over all 3^9 = 19,683 input matrices (taking about 390 million element-pair operations) is feasible in a few seconds on CPU. For the GPU kernel versions, the ternary domain bounds the verification cost in a way that float kernels never can.

The key insight is architectural: **ternary was built for compression, but it turns out to be a verification instrument**. The same property that makes ternary neural networks memory-efficient (2 bits per weight instead of 32) makes them verifiable. The state space is small enough to reason about formally, to enumerate exhaustively at small scales, and to provide tight theoretical bounds on output distributions at larger scales.

---

## IV. Physics-Based Invariants: Noether's Theorem at Compile Time

Emmy Noether's 1915 theorem establishes that every differentiable symmetry of the action of a physical system corresponds to a conserved quantity. Time-translation symmetry implies energy conservation. Spatial-translation symmetry implies momentum conservation. Rotational symmetry implies angular momentum conservation. The theorem is a bridge between symmetry (a structural property, discoverable through algebra) and conservation (a dynamical property, checkable through measurement).

The `ternary-noether` crate implements a discrete analogue of this theorem for ternary systems. The table of symmetries and conserved quantities reads:

| Symmetry | Discrete Generator | Conserved Quantity |
|---|---|---|
| Time translation | t → t + δ | Energy: E = Σ(p²/2 + x²/2) |
| Space translation | x → clamp(x + δ) | Momentum: P = Σ pᵢ |
| 90° Rotation | (x,y) → (-y,x) | Angular momentum: L = Σ(x·p_y − y·p_x) |
| Reflection(X) | (x,y) → (-x,y) | x-momentum |
| Reflection(Y) | (x,y) → (x,-y) | y-momentum |

The power of this for GPU verification is that conservation laws are **cheap to check and expensive to fake**. If a ternary kernel is supposed to implement a time-translation-invariant computation (as almost all stateless kernels are), then the energy of its ternary phase space must be constant across execution steps. If it drifts, something is wrong — either in the agent's intent or in the compilation. The `ternary-noether` crate's `Verification::verify_energy_conservation()` check computes this exactly, not approximately, because ternary values admit exact arithmetic.

The `ternary-hamiltonian` crate takes this further. Hamiltonian mechanics on the discrete phase space (q, p) ∈ {-1,0,+1}^2n uses symplectic integrators — specifically Störmer-Verlet (leapfrog) — that preserve the geometric structure of phase space. Liouville's theorem states that phase-space volume is conserved under Hamiltonian flow. In the discrete ternary case, this means the count of distinct occupied phase-space cells must be constant across time steps. The `LiouvilleTheorem::check_conservation()` function implements this check using `HashSet` sizes: an exact integer comparison, with no floating-point tolerance required.

What does this mean for compile-time verification? It means that a Flux bytecode program that claims to implement a Hamiltonian system can be checked — before compilation, before GPU dispatch — for the correct phase-space structure. The `flux-importer` (the bridge between Flux bytecode and cuda-oxide's MIR) could call `ternary-noether`'s verification infrastructure as a compilation pass. If the generated MIR does not preserve the declared symmetries of the kernel's intended computation, the compilation can be rejected with a precise diagnostic.

This is a *physics-based type system*. The type is not merely `Ternary` — it is `Ternary with Energy E conserved` or `Ternary with Momentum P = Σpᵢ`. These are stronger types than any existing programming language provides. They are the mathematical analogs of Rust's borrow checker: invariants enforced at compile time that prevent a whole class of runtime errors.

The ternary-electromagnetic system demonstrates this concretely. The `YeeLattice` implements discrete Maxwell equations with staggered E and B fields, leapfrog integration, and exact discrete charge conservation. The CFL stability condition (dt ≤ dx / (c·√2)) is a conservation-law-derived constraint on the kernel's time step. If an agent generates Flux bytecode for an electromagnetic simulation with a time step that violates CFL, a conservation-law checker can reject it at compile time, before the kernel ever reaches the GPU. No human needs to understand the PTX to know that the physics is wrong.

---

## V. The Z₃ Group Structure and What It Guarantees

The ternary set {-1, 0, +1} with the operations `tadd` and `tmul` forms a commutative ring with identity — specifically, it is isomorphic to Z₃, the cyclic group of order 3. This is not merely a curiosity about the arithmetic. It has deep implications for what properties ternary kernels can and cannot satisfy.

Z₃ is the simplest non-trivial cyclic group. Its group structure means that every element has an additive inverse (the negative), the group operation is associative and commutative, and the group has a single generator. In terms of verification, this means:

**Closure is exact, not approximate.** The fundamental problem with floating-point arithmetic in verified computing is that float arithmetic is not closed under any reasonable mathematical operations — overflow, underflow, and rounding mean that floating-point operations can leave the intended domain. Z₃ is closed under `tadd` and `tmul` by definition: `tmul(a, b) = clamp((a × b) mod 3, -1, 1)` never leaves {-1, 0, +1}. This makes the output type of every ternary operation provably bounded without any epsilon tolerance.

**The group structure implies specific algebraic identities** that a compiled kernel must satisfy. For any ternary values a, b, c: `tadd(a, tadd(b, c)) = tadd(tadd(a, b), c)` (associativity). `tadd(a, tneg(a)) = 0` (inverse). `tmul(a, 1) = a` (identity). These are theorems about Z₃, not heuristics. A verification harness can check them exhaustively for all 3^3 = 27 input triples.

**The cyclic structure of Z₃** means that 1 + 1 = -1 in ternary arithmetic (since 2 ≡ -1 mod 3). This is the rock-paper-scissors relationship: the three values form a dominance cycle. From `ternary-spiral`, this cyclic structure is what nucleates spiral waves — it is a dynamical consequence of the algebraic structure. A kernel that implements ternary dynamics will exhibit this cyclic behavior. A kernel that does not — say, because the agent generated Flux bytecode with incorrect overflow handling that uses saturation clamping instead of Z₃ arithmetic — will produce different spiral statistics. Statistical tests on the output distribution can detect this even without knowing the ground truth output.

This is a key verification strategy: **group-theoretic property testing**. For a ternary kernel that claims to implement Z₃ arithmetic, we can verify (a) algebraic closure by checking output types, (b) group axioms by exhaustive enumeration at small scale, and (c) dynamical signatures by checking that large-scale statistical properties of the output match the known Z₃ dynamics (spiral formation, 1:1:1 equilibrium distribution, etc.). None of these require reading the PTX. They require only the ability to run the kernel on controlled inputs and check the outputs against group-theoretic predictions.

---

## VI. Statistical Verification and the Oracle Problem

Even with the ternary state-space reduction, exhaustive verification breaks down for realistic kernel sizes. A ternary attention kernel with sequence length 512 and head dimension 64 involves 3^(512×64) possible inputs — a number so astronomically large that exhaustive testing is not even theoretically relevant. We need a different framework.

Statistical verification approaches the problem from a different angle: rather than testing all inputs, it tests whether the output *distribution* matches expectations. For ternary kernels, the expected output distributions are often known analytically.

Consider the `ternary-tnn` BitNet quantization: the weight quantization formula is `threshold_based(normalize(weight))`. For random Gaussian-distributed weights, the expected distribution of quantized trits is known: approximately 25% at -1, 50% at 0, and 25% at +1 when thresholded at ±0.5 standard deviations. A kernel that produces significantly different trit distributions on random inputs is doing something wrong. This is a chi-squared test: two degrees of freedom, straightforward to compute, no oracle needed.

The `ternary-spiral` RPS dynamics provide another statistical oracle. In a large RPS cellular automaton starting from a random 1:1:1 initialization, Shannon entropy H = -Σ pᵢ ln(pᵢ) should converge to ln(3) over time (maximum entropy for three states). Biodiversity metrics — Simpson index, Evenness — should stabilize near their maximum values. If a compiled spiral kernel produces entropy that diverges from ln(3), the compilation is wrong. The oracle is the known statistical physics of the RPS model, not a reference implementation.

This approach generalizes to a *kernel verification protocol* based on known statistical invariants:

1. **Distribution test**: For random ternary inputs, the output distribution should match analytical predictions.
2. **Entropy test**: For ergodic ternary systems, Shannon entropy should converge to the known maximum.
3. **Correlation test**: Ternary values at distance d should have known correlation structure (e.g., exponential decay for thermal systems, power law for critical systems).
4. **Symmetry test**: If the kernel claims a symmetry (rotation, reflection, time-reversal), statistical tests on the output should be invariant under that symmetry.

None of these tests require an oracle kernel that produces the ground-truth output. They test *structural properties of the output*, which are derivable from the mathematical structure of the computation. This is the key insight: **for physically grounded computations, the expected behavior is constrained by physics, not by a reference implementation**. Conservation laws, symmetry groups, and known statistical distributions provide an independent ground truth that does not require another agent to generate it.

---

## VII. Formal Methods and the Lean/Coq Interface

The formal verification literature offers tools that, while primarily designed for human-authored programs, can be adapted to agent-generated code through the ternary state-space reduction.

**Coq** and **Lean** are interactive theorem provers that can, in principle, prove properties of programs by construction. The challenge for GPU kernels is representing PTX semantics in a proof assistant's type theory. This is non-trivial: PTX has explicit memory address spaces, warp-level synchronization primitives, and non-deterministic warp scheduling. A full formalization of PTX semantics in Lean would be a multi-year research project.

However, the ternary ecosystem offers a shorter path. The core operations — `tadd`, `tmul`, the Z₃ group axioms — can be formalized in Lean 4 in a few hundred lines, and the key theorems (associativity, commutativity, conservation laws) can be proved once and used forever. The interesting engineering question is: how much of the verification burden can be discharged at the *Flux bytecode level* rather than the *PTX level*?

The answer is: much more than you might think. If the Flux bytecode type system is strong enough to enforce ternary closure (all operations preserve the {-1,0,+1} invariant), then the correctness of the compiled PTX follows from (a) the correctness of the Flux semantics, proved in Lean, and (b) the correctness of the cuda-oxide compiler, verified by a separate Rust verification effort. This is the *compositional verification* approach: prove each layer correct and compose the proofs.

**TLA+** (Temporal Logic of Actions) is more appropriate for the distributed aspects of the system — the SmartCRDT state synchronization, the agent assignment OR-sets, the kernel hotswap protocol. TLA+ specifications describe what states the system can be in and what transitions are allowed. For the cudaclaw persistent kernel runtime, a TLA+ spec could verify that the CRDT merge protocol never produces a state where two agents hold the same GPU assignment simultaneously. This is exactly the kind of distributed correctness property that statistical testing cannot easily verify.

**SMT solvers** (Z3, CVC5) provide bounded verification for finite domains. For ternary kernels with bounded input size, an SMT formulation can decide whether the kernel satisfies a property for all inputs of size up to N. For N=8 (8 trits = 1 byte), this is immediate. For N=16, SMT solvers running on modern hardware can typically handle the constraint satisfaction in seconds. Beyond N≈24, the exponential blowup exceeds current solver capabilities, but this still covers a useful range of small kernels.

The specific coupling to the Flux→PTX pipeline would look like this: the `oxide-constructs` crate maintains a *verification certificate* alongside each compiled PTX artifact. The certificate is a tuple (kernel_hash, properties_verified, verification_method, timestamp). Properties might include: "Z₃ closure verified by SMT for inputs up to N=16", "energy conservation verified analytically for all inputs", "output distribution verified statistically on 10M random inputs with p < 0.001". The certificate does not prove the kernel is correct — it proves that certain properties hold. A human (or a higher-level verification agent) can then decide whether those certified properties are sufficient for the intended use.

---

## VIII. Runtime Verification: cudaclaw's Warp-Level Invariant Checking

Compile-time verification answers the question "does this kernel *seem* correct?" Runtime verification answers the question "is this kernel *behaving* correctly *right now*?"

The cudaclaw persistent kernel runtime provides infrastructure for warp-level invariant checking during execution. The mechanism described in the architectural analysis is a commitment scheme: each warp independently computes a cryptographic commitment to its output, and when all warps in a block commit, the block's aggregated commitment is compared against an expected commitment derived from the Flux graph.

For ternary kernels, a simpler and cheaper invariant is available: **type invariant checking**. Since all values should remain in {-1, 0, +1} throughout the computation, any deviation is immediately detectable. A warp that produces a value of 2 or -2 has violated the ternary invariant, indicating either a compiler bug (incorrect Z₃ arithmetic), a hardware fault (soft error flipping a bit), or an agent intent error (the bytecode uses integer addition instead of Z₃ addition). The check is a single `vmin/vmax` operation per output element — essentially free at warp scale.

**Conservation law monitoring** is more expensive but more powerful. If the kernel implements a Hamiltonian system, the cudaclaw runtime can maintain a running sum of the energy observable (Σ(p²/2 + x²/2) over all phase-space coordinates). This sum should be constant — or bounded by a known small drift for near-conservative systems. If it exceeds a threshold, the kernel is aborted and the error propagated to the agent. The `ternary-hamiltonian` crate's `verify_energy_conservation()` function already implements this check; the integration into cudaclaw would expose it as a warp-level callback.

The **SmartCRDT commitment mechanism** provides probabilistic guarantees for non-Hamiltonian kernels. For a kernel with N warps, each warp computes a 128-bit commitment to its intermediate state at a designated checkpoint. When all warps have committed, the commitments are XOR-aggregated (this preserves the XOR-commit property: the aggregate is a commitment to the XOR of all states). The expected aggregate commitment can be pre-computed from the Flux graph if the input is known, or compared against a reference execution on a small-scale simulation. The probability of an undetected error given 128-bit commitments is below 2^(-128) — cryptographically negligible.

The performance cost of runtime verification is bounded. For conservation law checking: one reduction per timestep, O(N) in the number of phase-space coordinates. For type invariant checking: one clamp+compare per output element, essentially a no-op at warp scale. For commitment-based checking: one hash per warp per checkpoint, approximately 5-10 cycles per warp on modern GPUs. For a kernel executing at 400,000 operations per second with 10,000 agents, the verification overhead is on the order of 1-5% of total kernel runtime — a reasonable tradeoff for catching semantically wrong kernels before their errors propagate.

---

## IX. lever-runner and fastloop-guard as the Last Line of Defense

The lever-runner and fastloop-guard components occupy a privileged position in the Flux→PTX pipeline: they are the boundary between the compilation/verification world and the execution world. lever-runner translates agent intent into GPU commands. fastloop-guard validates those commands before dispatch, operating at sub-millisecond latency.

This is the correct architecture for a *last-line-of-defense* verification layer. By the time a command reaches fastloop-guard, it has already survived Flux bytecode generation, type checking, conservation law verification at compile time, and possibly statistical property testing. The guard's job is not to re-verify correctness — it is to enforce *safety bounds* that prevent any failure mode from having unbounded consequences.

The guard's checks, applied at sub-millisecond latency, should include:
- **Memory bound checks**: No kernel command should access GPU memory outside its allocated sandbox region. This catches buffer overflows and out-of-bounds access before they corrupt other kernels' state.
- **Execution time limits**: A kernel that runs longer than its declared maximum execution time is likely deadlocked or in an infinite loop. The guard aborts it.
- **Resource consumption limits**: Maximum register count, shared memory usage, and warp occupancy are declared in the construct manifest. A kernel that exceeds these limits at load time (before execution) is rejected.
- **Rate limiting**: If the same kernel generates errors at a rate above a threshold (more than 0.01% of executions producing out-of-domain values), the guard quarantines it and notifies the agent.

The combination of fastloop-guard's safety bounds with the upstream verification stack creates a *defense-in-depth* architecture. No single layer needs to be complete. Compile-time conservation law checking catches most systematic intent errors. Runtime invariant monitoring catches hardware faults and corner-case bugs that slipped through compile-time checking. fastloop-guard prevents any surviving errors from causing unbounded damage. The agent can be notified at any layer and can regenerate the Flux bytecode with the verification error as a feedback signal.

---

## X. What a Verified Compilation Pipeline Would Actually Look Like

Assembling the pieces above into a coherent pipeline requires addressing a fundamental tension: comprehensive verification is expensive, and the Flux→PTX system needs to operate at sub-10ms compile-deploy-execute cycles for interactive workloads.

The resolution is *tiered verification*: not every kernel requires every verification step, and the appropriate verification depth depends on the kernel's risk profile.

**Tier 0: Basic Type Safety (always applied, sub-millisecond)**
- Z₃ closure check: all operations produce values in {-1, 0, +1}
- Memory access bounds: all loads/stores within declared allocation
- Barrier placement: no unpaired `SYNC_THREADS` instructions
- These are implemented as O(n) passes in `flux-importer`

**Tier 1: Algebraic Property Verification (applied to new kernels, ~10ms)**
- Group axiom verification by exhaustive enumeration for inputs up to K=8
- Z₃ arithmetic correctness: all 27 input triples satisfy associativity, commutativity, distributivity
- Implemented as a test harness that runs on CPU alongside compilation

**Tier 2: Conservation Law Checking (applied when kernel declares physics, ~100ms)**
- Energy conservation via `ternary-noether` before compilation
- Momentum and angular momentum checks
- Liouville phase-space volume check
- Implemented as a pass in the `oxide-constructs` compilation pipeline

**Tier 3: Statistical Property Verification (applied to high-value kernels, ~1s)**
- Distribution tests on 10^6 random inputs
- Entropy convergence tests for ergodic systems
- Symmetry invariance tests for declared symmetries
- Implemented as a background verification job; kernel is marked "statistically verified" when complete

**Tier 4: Formal Verification Certificate (applied to safety-critical kernels, minutes)**
- SMT-based bounded verification for inputs up to K=16
- TLA+ specification check for distributed behavior
- Lean proof of key algebraic properties
- Implemented as an offline verification service; certificate stored in `oxide-constructs` cache

At runtime, cudaclaw applies:
- Continuous type invariant monitoring (free)
- Periodic conservation law checks (1-5% overhead)
- SmartCRDT commitment checking (1-5% overhead)

fastloop-guard enforces:
- Memory bounds at dispatch (sub-millisecond)
- Execution time limits (ongoing)
- Error rate quarantine (reactive)

The verification certificate stored with each compiled PTX artifact documents which tiers were applied and what properties were verified. Downstream consumers of the kernel can decide whether the certificate meets their requirements. An interactive agent querying a low-stakes kernel might accept Tier 0-1 certification. A fleet running safety-critical compute might require Tier 0-4 certification with active runtime monitoring.

---

## XI. The Intent Error Problem and Its Partial Resolution

Nothing in the above pipeline resolves the deepest form of the verification gap: the case where the agent's intent itself is wrong. A kernel can be type-safe, algebraically correct, energy-conserving, statistically plausible, and formally verified against its own specification — and still be computing the wrong thing, because the specification was wrong.

This is not a theoretical edge case. It is the default condition for agent-generated code. The agent does not have a ground truth to compare against. It has a training distribution, an inference prompt, and a sampling strategy. The generated Flux bytecode is a hypothesis about what the correct computation looks like. The verification pipeline above tests whether the kernel is *internally consistent* — consistent with ternary arithmetic, consistent with declared conservation laws, consistent with known statistical properties. Internal consistency is necessary but not sufficient for external correctness.

The partial resolution available within the system architecture involves three components:

**Differential execution**: Run the same Flux bytecode on two independent implementations — the cudaclaw GPU kernel and a CPU-side reference implementation using the `ternary-core` crate's pure-Rust operations. Both implementations compute on the same input; discrepancies indicate either a compilation error (the GPU kernel differs from the Flux semantics) or an agent intent error that manifests as semantic deviation from the pure-Rust baseline. The pure-Rust implementation cannot tell us whether the intent is *right*, but it can tell us whether the *compilation preserved the intent*. This converts the intent error detection problem into a differential testing problem, which is at least mechanically addressable.

**Reference kernel comparison**: The `oxide-constructs` crate's content-addressed cache maps Flux graph hashes to verified PTX artifacts. When an agent generates a Flux program that is structurally similar (via fuzzy hashing or graph edit distance) to a previously verified kernel, the new kernel can be compared against the reference. If the behavioral profiles diverge significantly, the agent is notified. This creates a *corpus of verified behaviors* that grows over time and provides an increasingly comprehensive reference against which new agent-generated kernels can be checked.

**The oracle bootstrapping problem**: Ultimately, the oracle for "is this computation the right computation?" must come from outside the system — from the task-level evaluation that tells the agent whether its outputs were useful. In the language of reinforcement learning, verification at the kernel level is verification of the *policy function*, while task-level evaluation is verification of the *value function*. Both are necessary. The Flux→PTX pipeline provides rich feedback for the policy (this kernel is type-unsafe, this kernel violates energy conservation, this kernel differs from its reference). The task-level evaluation provides feedback for the value (this computation produced useful results downstream). Together they form a complete feedback loop.

---

## XII. Conclusion: The Verification Stack as a Research Program

The verification gap in the Flux→PTX pipeline is not a single problem but a family of related problems, each addressable by different tools, each requiring different tradeoffs between completeness and cost.

What the ternary ecosystem offers that no prior GPU framework has offered is a *physically grounded type system* that makes many of these problems tractable. The {-1, 0, +1} domain is small enough for exhaustive testing of small kernels, provably closed under Z₃ arithmetic, aligned with Noether's conservation laws, and amenable to known statistical characterization. These properties are not incidental — they are why the ternary abstraction is the right foundation for agent-generated GPU code. Not because it compresses well (though it does) or because it maps efficiently to hardware (though it does). But because its bounded, algebraically structured domain transforms the open problem of GPU kernel verification into a collection of solvable subproblems.

The pipeline that emerges from taking verification seriously looks like this: compile-time conservation law checking via `ternary-noether` and `ternary-hamiltonian`; algebraic property verification via Z₃ group theory; statistical property testing via known entropy and distribution invariants; SMT-based bounded formal verification for small kernels; runtime monitoring via cudaclaw's warp-level conservation checks and SmartCRDT commitments; and last-line-of-defense safety bounds via fastloop-guard.

No single layer of this stack is complete. Together they implement a defense-in-depth that narrows the verification gap from "we have no idea if this is right" to "we have verified these specific properties, and we have detected no violations in 10^8 executions." That is not a proof of correctness. But it is the best available answer to the question: *how do you trust code that no human wrote?*

The honest answer is: carefully, incrementally, and with the mathematics on your side.

---

*References: `ternary-noether` (conservation law verification), `ternary-hamiltonian` (symplectic integration, Liouville's theorem), `ternary-core` (Z₃ ring axioms, exhaustive verification), `ternary-spiral` (RPS dynamics, biodiversity metrics, Shannon entropy), `ternary-diehard` (fault tolerance, population statistics), `ternary-tnn` (BitNet quantization, STE gradient), `ternary-tensor` (matmul with ternary clamping, CP decomposition), cuda-oxide (MIR→Pliron→NVVM→PTX pipeline), cudaclaw (persistent kernels, warp-level consensus), fastloop-guard (sub-ms validation, rate limiting), SmartCRDT (128-bit commitment scheme, vector clocks), `oxide-constructs` (content-addressed PTX cache, verification certificates).*
