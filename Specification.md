# AETHEROS: Architecture and Design Intent Specification

**Version:** 2.0.0
**Status:** Consolidated
**Last Updated:** 2025-12-26

---

## 1. Executive Summary

AETHEROS is a **quadripartite microkernel operating system** designed from first principles for modern heterogeneous computing platforms. It represents a fundamental rethinking of operating system architecture, guided by the Japanese aesthetic principle of **MA (間)**—meaningful negative space.

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER SPACE                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Applications│  │   SI Agents │  │  Services   │  │   Drivers   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CAPABILITY-BASED SYSTEM CALL INTERFACE                   │
│                  (All operations require capability tokens)                  │
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUADRIPARTITE KERNEL                                 │
│                                                                              │
│  ┌───────────────────┐              ┌───────────────────┐                   │
│  │   GOVERNANCE (司)  │◄────────────►│   COGNITIVE (理)   │                   │
│  │   Constitutional   │              │   Reasoning &     │                   │
│  │   Authority        │              │   Scheduling      │                   │
│  │   ───────────────  │              │   ───────────────  │                   │
│  │   • Policy         │              │   • Planning      │                   │
│  │   • Invariants     │              │   • GPU/NPU mgmt  │                   │
│  │   • Audit          │              │   • Optimization  │                   │
│  └─────────┬─────────┘              └─────────┬─────────┘                   │
│            │                                  │                              │
│            ▼                                  ▼                              │
│  ┌───────────────────┐              ┌───────────────────┐                   │
│  │   PHYSICAL (体)    │◄────────────►│   EMOTIVE (気)     │                   │
│  │   Hardware         │              │   User Advocacy   │                   │
│  │   Abstraction      │              │   & Experience    │                   │
│  │   ───────────────  │              │   ───────────────  │                   │
│  │   • Memory         │              │   • Intent model  │                   │
│  │   • Interrupts     │              │   • Priority      │                   │
│  │   • I/O, Timing    │              │   • Quality       │                   │
│  └───────────────────┘              └───────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDWARE ABSTRACTION LAYER                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ CPU x96  │  │ DDR5 3TB │  │ GPU RDNA │  │ NPU XDNA │  │ NVMe/PCIe│      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Innovations

| Innovation | Description | Benefit |
|------------|-------------|---------|
| **Quadripartite Architecture** | Four specialized kernels (Governance, Physical, Emotive, Cognitive) | Clear separation of concerns; single responsibility per kernel |
| **Capability-Based Security** | Unforgeable tokens replace ambient authority | No confused deputy; all access explicitly authorized |
| **User Primacy Invariant** | Human authority is constitutional; cannot be overridden | System serves user, never the reverse |
| **Emotive Kernel** | Dedicated kernel for user intent inference and experience quality | System anticipates and optimizes for user goals |
| **Dataflow Execution** | Computation as dependency graphs, not threads | Natural parallelism; deterministic scheduling |
| **Typed Channels** | Session-typed inter-kernel communication | Protocol correctness verified at compile time |
| **Tripartite Network Channels** | Internal/Private/Public isolation | System actor never receives from public channels |

### 1.3 Target Platform

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Threadripper PRO 9995WX (96 cores, 192 threads, Zen 5) |
| **Memory** | 3TB DDR5 (12-channel, ~460 GB/s bandwidth) |
| **GPU** | Integrated RDNA (HIP/ROCm programmable) |
| **NPU** | XDNA AI Engine (inference acceleration) |
| **Storage** | NVMe (RAM-resident execution; storage for checkpoints) |

### 1.4 Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Cold Boot** | < 2 seconds | Bare-metal initialization to kernel entry |
| **Warm Boot** | < 800 ms | Checkpoint restore to full operation |
| **IPC Latency** | < 1.5 μs (topology-dependent) | Competitive with L4/seL4; see Section 11.4 |
| **Context Switch** | < 1 μs | Minimal overhead |
| **End-to-End User Action** | < 16 ms | Soft real-time responsiveness |

### 1.5 Actor Model

```
Human (Constitutional Authority)
   │
   ├──► System (Maintenance & Infrastructure)
   │       • Cannot act against Human interests
   │       • Manages resources on Human's behalf
   │
   └──► SI - Synthetic Intelligence (Delegated Authority)
           • Authority explicitly delegated by Human
           • Can be constrained or revoked at any time
           • Cannot exceed delegator's authority
```

### 1.6 Formal Verification Strategy

| Layer | Tool | Purpose |
|-------|------|---------|
| **Behavioral** | TLA+ | State machine correctness, liveness, safety |
| **Structural** | Lean 4 | Type-level proofs, capability algebra |
| **Protocol** | Session Types | Communication protocol correctness |

### 1.7 Primary Language

**Rust**, with strategic use of C/C++ for AMD ecosystem integration and assembly for hardware primitives.

---

## 2. Requirement Traceability Matrix

All requirements use format: `REQ-[CATEGORY]-[NUMBER]`

### 2.1 Categories

| Category | Description |
|----------|-------------|
| **ARCH** | Architectural requirements |
| **SEC** | Security requirements |
| **PERF** | Performance requirements |
| **CAP** | Capability system requirements |
| **KERN** | Kernel requirements |
| **IPC** | Inter-process/kernel communication |
| **MEM** | Memory management |
| **PROC** | Process model |
| **FORM** | Formal verification |

### 2.2 Core Requirements

| ID | Requirement | Priority | Verification |
|----|-------------|----------|--------------|
| **REQ-ARCH-001** | System SHALL implement quadripartite kernel architecture | CRITICAL | Design review, TLA+ model |
| **REQ-ARCH-002** | Each kernel SHALL have single, well-defined responsibility | CRITICAL | Code review, architecture audit |
| **REQ-ARCH-003** | Kernels SHALL communicate only through typed channels | CRITICAL | Session type verification |
| **REQ-SEC-001** | All resource access SHALL require valid capability | CRITICAL | Lean 4 proof, fuzzing |
| **REQ-SEC-002** | Capabilities SHALL be unforgeable | CRITICAL | Lean 4 no-forgery proof |
| **REQ-SEC-003** | Capability authority SHALL NOT be amplifiable | CRITICAL | Lean 4 no-amplification proof |
| **REQ-SEC-004** | Capability revocation SHALL propagate to all derived capabilities | HIGH | TLA+ model checking |
| **REQ-SEC-005** | Human actor SHALL have irrevocable constitutional authority | CRITICAL | Governance kernel invariant |
| **REQ-SEC-006** | System actor SHALL NEVER receive from public network channels | CRITICAL | Channel routing verification |
| **REQ-PERF-001** | Warm boot time SHALL be less than 800ms | HIGH | Benchmark |
| **REQ-PERF-002** | Cold boot time SHALL be less than 2 seconds | MEDIUM | Benchmark |
| **REQ-PERF-003a** | Same-core IPC latency SHALL be less than 500 nanoseconds | HIGH | Benchmark |
| **REQ-PERF-003b** | Cross-CCX IPC latency SHALL be less than 1.2 microseconds | HIGH | Benchmark |
| **REQ-PERF-003c** | Cross-CCD IPC latency SHALL be less than 1.5 microseconds | HIGH | Benchmark |
| **REQ-PERF-004** | Context switch time SHALL be less than 1 microsecond | HIGH | Benchmark |
| **REQ-PERF-005** | End-to-end user action latency SHALL be less than 16ms | HIGH | Benchmark |
| **REQ-CAP-001** | Capabilities SHALL encode resource identity + rights | CRITICAL | Type definition |
| **REQ-CAP-002** | Rights lattice SHALL support intersection and subset operations | HIGH | Lean 4 proof |
| **REQ-CAP-003** | Capability delegation SHALL NOT exceed delegator's authority | CRITICAL | Lean 4 proof |
| **REQ-KERN-001** | Governance kernel SHALL enforce constitutional invariants | CRITICAL | Formal verification |
| **REQ-KERN-002** | Physical kernel SHALL abstract all hardware access | HIGH | Code review |
| **REQ-KERN-003** | Emotive kernel SHALL maintain user intent model | HIGH | Behavioral testing |
| **REQ-KERN-004** | Cognitive kernel SHALL schedule all computation | HIGH | Scheduler verification |
| **REQ-IPC-001** | Inter-kernel channels SHALL be session-typed | HIGH | Type checking |
| **REQ-IPC-002** | Channel communication SHALL preserve capability integrity | CRITICAL | Lean 4 proof |
| **REQ-MEM-001** | Memory domains SHALL provide isolation between protection domains | CRITICAL | Hardware + software verification |
| **REQ-MEM-002** | Memory allocation SHALL be capability-controlled | HIGH | Code review |
| **REQ-MEM-003** | OOM conditions SHALL be handled gracefully without panic | HIGH | Stress testing |
| **REQ-PROC-001** | Process creation SHALL require appropriate capability | HIGH | Code review |
| **REQ-PROC-002** | Process termination SHALL release all held resources | CRITICAL | Resource tracking |
| **REQ-PROC-003** | Exceptions SHALL be delivered via capability-controlled channels | HIGH | Design review |
| **REQ-FORM-001** | Core capability algebra SHALL be proven in Lean 4 | CRITICAL | Proof compilation |
| **REQ-FORM-002** | System state machine SHALL be modeled in TLA+ | HIGH | Model checking |
| **REQ-FORM-003** | All safety properties SHALL pass TLC model checking | HIGH | TLC output |

---

## 3. Philosophical Foundations

### 3.1 The Problem with Existing Operating Systems

Current operating systems carry accumulated assumptions from decades past:

| Legacy Assumption | Modern Reality |
|---|---|
| CPU is the compute center | CPU is the orchestrator; GPU/NPU perform heavy computation |
| Memory is scarce; disk is storage | Memory is abundant; persistent memory blurs boundaries |
| Processes are isolated kingdoms | Fine-grained sharing with strong isolation required |
| Sequential by default | Parallel by default; sequential is the special case |
| Files are the universal abstraction | Typed, structured data demands richer primitives |
| Security bolted on (DAC/MAC) | Security must be architectural (capabilities) |
| User adapts to system | System serves user |

AETHEROS rejects these assumptions. It is a fresh start.

### 3.2 MA (間): Meaningful Negative Space

The system is defined as much by what it _excludes_ as what it includes. Every system call, every abstraction, every capability exists because its absence would create a hole. Nothing ornamental. Nothing vestigial.

> We shape clay into a pot, but it is the emptiness inside that holds whatever we want.
> — *Lao Tzu*

**Foundational Inspirations:**

> Conceptual [functional] simplicity, structural complexity achieves a greater state of humanity.
> — *Junctions*

> Expose and elucidate the GUIDs move behind labels.
> — *DavidSlight*

Every abstraction in AETHEROS reveals its underlying identity on demand. Labels are conveniences; GUIDs are truth.

**The Three-Tome Model:**

Information architecture follows three insulated layers:
- **Interface** — What you see
- **Implementation** — How it works
- **Intent** — Why it exists

Each tome is insulated from the others, enabling independent evolution while preserving coherent consumption.

**The MA Imperative:**

| Principle | Meaning | Violation Example |
|-----------|---------|-------------------|
| **Form follows Function** | Structure exists to serve purpose, never the reverse | UI that forces workflow; API that constrains capability |
| **Function serves USER** | All functionality ultimately serves Human, SI, or System | Features that serve vendor metrics over user goals |
| **Transparency by Default** | Nothing hidden; all state observable; all decisions auditable | Hidden telemetry; opaque algorithms; undocumented behavior |
| **No Lock-In** | User data portable; formats open; dependencies replaceable | Proprietary formats; vendor-specific APIs; closed ecosystems |
| **No Hidden Agendas** | System acts solely for authorized actors; no dark patterns | Nudges toward vendor benefit; artificial limitations; upselling |

```lean
-- MA principle: form must serve function, function must serve user
-- REQ-ARCH-002: Single responsibility per component
structure MACompliance where
  formFollowsFunction    : Bool  -- Structure enables purpose
  functionServesUser     : Bool  -- Purpose benefits an authorized actor
  transparentByDefault   : Bool  -- State and decisions are observable
  noVendorLockIn         : Bool  -- Data portable, formats open
  noHiddenAgendas        : Bool  -- No dark patterns or hidden motives

def isMACompliant (c : Component) : Bool :=
  c.maCompliance.formFollowsFunction ∧
  c.maCompliance.functionServesUser ∧
  c.maCompliance.transparentByDefault ∧
  c.maCompliance.noVendorLockIn ∧
  c.maCompliance.noHiddenAgendas

theorem ma_compliance_required (c : Component) :
    c ∈ SystemComponents → isMACompliant c := by
  intro h_in_system
  exact component_verified_at_integration c h_in_system
```

### 3.3 User Primacy

> **REQ-SEC-005**: Human actor SHALL have irrevocable constitutional authority

The Human is the ultimate authority. The system exists to serve human goals, not the reverse. This is a constitutional invariant enforced by the Governance kernel.

```lean
-- User Primacy as constitutional law
axiom human_primacy : ∀ (action : SystemAction),
  action.beneficiary ≠ Actor.human →
  action.doesNotHarm Actor.human

theorem user_primacy_irrevocable :
    ∀ (config : SystemConfiguration),
      config.respectsUserPrimacy := by
  intro config
  exact governance_enforces_primacy config
```

### 3.4 Tripartite Actor Model

| Actor | Role | Authority Source | Constraints |
|-------|------|------------------|-------------|
| **Human** | Constitutional authority | Inherent (axiomatic) | None (supreme) |
| **System** | Infrastructure maintenance | Delegated by design | Cannot contradict Human |
| **SI (Synthetic Intelligence)** | Delegated cognitive tasks | Explicitly delegated by Human or System | Cannot exceed delegator's authority |

**Authority Hierarchy**: Human > SI (acting for Human) > System > SI (acting for System)

```lean
-- REQ-SEC-005: Actor hierarchy
inductive Actor where
  | human    : UserId → Actor      -- Constitutional authority
  | system   : ServiceId → Actor   -- Infrastructure (delegated)
  | si       : AgentId → Actor     -- AI agents (delegated)
  deriving DecidableEq, Repr

def Actor.hasAuthorityOver : Actor → Actor → Bool
  | .human _, .system _ => true
  | .human _, .si _ => true
  | .human u1, .human u2 => u1 = u2  -- Only over self
  | _, .human _ => false              -- Nothing over human
  | .system _, .si _ => false         -- SI reports to Human, not System
  | .si _, .system _ => false         -- Incomparable
  | _, _ => false

theorem human_authority_supreme :
    ∀ (h : Actor) (a : Actor),
      h.isHuman → h ≠ a → ¬(a.hasAuthorityOver h) := by
  intro h a h_is_human h_neq
  cases a with
  | human _ => simp [Actor.hasAuthorityOver]
  | system _ => simp [Actor.hasAuthorityOver]
  | si _ => simp [Actor.hasAuthorityOver]
```

### 3.5 Mathematical Consistency

The system is an algebraic object: operations compose, invariants hold, state transitions are total functions. No undefined behavior. No "it depends." The specification _is_ the system; implementation merely instantiates it.

### 3.6 Transparency Invariants

| Transparency Domain | What Is Visible | How |
|---------------------|-----------------|-----|
| **State** | All system state observable by appropriate actors | State query APIs, debug introspection |
| **Decisions** | All policy decisions auditable with reasoning | Audit log with decision traces |
| **Data Flow** | All data movement trackable | Capability-based data lineage |
| **Dependencies** | All external dependencies explicit | Manifest with versions, sources |
| **Telemetry** | All telemetry visible and controllable by user | Telemetry dashboard, opt-in only |

```lean
-- Transparency invariant: every state change has observable justification
theorem transparency_invariant (s s' : SystemState) (t : Transition) :
    t s = s' →
    ∃ j : Justification, observable j ∧ explains j t

-- No dark patterns: every user-facing action has clear purpose
def isDarkPattern (action : UserAction) : Bool :=
  action.hiddenConsequences.nonEmpty ∨
  action.misleadingPresentation ∨
  action.benefitsVendorOverUser

theorem no_dark_patterns :
    ∀ action : UserAction, ¬ isDarkPattern action

-- Vendor lock-in elimination: all data formats are open
axiom data_portability :
    ∀ d : UserData, ∃ f : OpenFormat, exportable d f
```

---

## 4. Architectural Overview

### 4.1 The Quadripartite Model

| Kernel | Symbol | Question Answered | Time Horizon | Nature |
|--------|--------|-------------------|--------------|--------|
| **Governance** | Nous/司 | What is permitted? What must hold? | Eternal | Normative, Invariant |
| **Physical** | Soma/体 | What is? What happened? | Immediate | Factual, Reactive |
| **Emotive** | Thymos/気 | What matters to this user right now? | Near-term | Evaluative, Subjective |
| **Cognitive** | Logos/理 | What should we do? | Strategic | Deliberative, Objective |

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GOVERNANCE KERNEL                           │
│                            (Nous/司)                                │
│   Policy • Invariants • Arbitration • Oversight                     │
│   "What is permitted? What must always hold?"                       │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         │ policy             │ policy             │ policy
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    COGNITIVE    │  │    EMOTIVE      │  │    PHYSICAL     │
│     KERNEL      │  │     KERNEL      │  │     KERNEL      │
│    (Logos/理)   │  │   (Thymos/気)   │  │   (Soma/体)     │
│                 │  │                 │  │                 │
│  Reasoning      │  │  User Advocacy  │  │  Actuation      │
│  Planning       │  │  Prioritization │  │  Sensation      │
│  Inference      │  │  Experience QoS │  │  Memory/IO      │
│  Optimization   │  │  Intent Model   │  │  Timing         │
│                 │  │                 │  │                 │
│  DELIBERATIVE   │  │  SUBJECTIVE     │  │  REACTIVE       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Why Four Kernels?**

This pattern recurs across philosophy and systems theory:

| Tradition | Physical | Emotive | Cognitive | Governance |
|---|---|---|---|---|
| Platonic Soul | Appetite | Spirit (Thymos) | Reason (Logos) | The Good |
| Jungian Functions | Sensation | Feeling | Thinking | Self |
| OODA Loop | Observe | Orient | Decide | Act (oversight) |
| Viable System Model | Operations | Coordination | Control | Policy |

### 4.2 Key Innovations

#### 4.2.1 Capabilities, Not Permissions

Every resource reference is an unforgeable token encoding _what you can do_. No ambient authority. No access control lists consulted at runtime.

```
Traditional Model:
    Process → "Can I access /dev/gpu0?" → Kernel checks ACL → Yes/No

AETHEROS Model:
    Process holds Capability<GPU, ReadWrite> → Access is the capability
    No runtime check needed; possession IS authorization
```

#### 4.2.2 Compute as Dataflow, Not Threads

Computation is expressed as dependency graphs:

```
[Sensor Input] → [Preprocess:NPU] → [Inference:GPU] → [Decision:CPU] → [Output]
```

The Cognitive kernel schedules across heterogeneous compute units.

#### 4.2.3 Memory Domains with Explicit Coherence

| Coherence | Semantics |
|---|---|
| *coherent* | Hardware-maintained (CPU RAM, CXL Type 3) |
| *explicit* | Requires sync primitives (GPU VRAM, shared buffers) |

Persistence is orthogonal—any domain can have `persistent: true`.

#### 4.2.4 Everything is a Typed Channel

Not "everything is a file" (too unstructured) or "everything is an object" (too heavyweight). AETHEROS uses typed, capability-protected channels with schema evolution:

```lean
-- Channels are typed; protocol violations are compile errors
def sensorChannel : Channel SensorReading := ...

-- Session types ensure correct sequencing
session GPUJob where
  client sends   : ComputeGraph
  server sends   : Placement
  client sends   : Confirm
  server sends   : JobHandle
```

#### 4.2.5 Tripartite Network Channels

All network communication is classified into three isolated channel classes:

| Channel | Purpose | Security |
|---------|---------|----------|
| **Internal Local** | System telemetry, health monitoring | Highest priority, physically isolated partition |
| **Private** | User-initiated trusted communications | Encrypted, user-approved endpoints |
| **Inbound Public** | External internet traffic | NEVER routes to System actor |

```
Priority: Internal (1) > Private (2) > Public (3)

Critical invariant: System actor NEVER receives from public channels
                    Internal partition NEVER shares resources with public
```

### 4.3 System Layers

```
┌────────────────────────────────────────────────────────────────┐
│                      User Space                                │
│    Applications, Services, Shells                              │
│    (Language-agnostic, capability-constrained)                 │
├────────────────────────────────────────────────────────────────┤
│                   Compute Fabric Runtime                       │
│    Unified scheduling across CPU/GPU/NPU                       │
│    Dataflow graphs, not threads                                │
│    Memory: explicit coherence domains, zero-copy channels      │
├────────────────────────────────────────────────────────────────┤
│                    Microkernel Core                            │
│    Four kernels: Governance, Emotive, Cognitive, Physical      │
│    Capabilities, IPC, resource accounting                      │
│    Minimal TCB (~15-20K lines target)                          │
│    Formally verifiable where possible                          │
├────────────────────────────────────────────────────────────────┤
│                   Hardware Abstraction                         │
│    CPU (Zen 5), GPU (RDNA), NPU (XDNA)                         │
│    Unified address space with explicit domains                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Foundational Primitives

### 5.1 The Three Primitives

AETHEROS is built on exactly three foundational concepts:

| Primitive | Definition | Role |
|-----------|------------|------|
| **Capability** | Unforgeable token encoding authority | Access control, security |
| **State** | Typed, versioned, observable data | Computation substrate |
| **Transition** | Pure function from state to state | All change |

Everything else derives from these three.

```lean
-- The three foundational primitives
-- REQ-CAP-001: Capability encodes resource + rights

-- 1. CAPABILITY: Authority to act
structure Capability (Resource : Type) where
  resource   : Resource        -- What it grants access to
  rights     : Rights          -- What operations are permitted
  provenance : Provenance      -- Derivation chain (for revocation)
  epoch      : Epoch           -- Version for revocation checking
  deriving DecidableEq, Repr

-- 2. STATE: The substrate of computation
structure State where
  domains     : DomainMap       -- Isolated state containers
  channels    : ChannelMap      -- Communication conduits
  schedule    : ScheduleState   -- What's running where
  epoch       : Epoch           -- Global version
  invariants  : InvariantSet    -- What must hold

-- 3. TRANSITION: The only way state changes
structure Transition where
  precondition  : State → Bool           -- When can this fire?
  effect        : State → State          -- What does it do?
  postcondition : State → State → Bool   -- What must be true after?

-- Fundamental theorem: transitions preserve invariants
theorem transition_preserves_invariants (t : Transition) (s : State) :
    t.precondition s →
    s.invariants.allHold s →
    s.invariants.allHold (t.effect s) := by
  intro h_pre h_inv
  exact t.postcondition_implies_invariants s (t.effect s) h_pre h_inv
```

### 5.2 Rights Algebra

Rights form a bounded lattice with well-defined operations:

```lean
-- Rights as a lattice
-- REQ-CAP-002: Rights support intersection and subset
structure Rights where
  bits : UInt64
  deriving DecidableEq, Repr

namespace Rights
  def none     : Rights := ⟨0⟩
  def read     : Rights := ⟨1⟩
  def write    : Rights := ⟨2⟩
  def execute  : Rights := ⟨4⟩
  def grant    : Rights := ⟨8⟩
  def revoke   : Rights := ⟨16⟩
  def all      : Rights := ⟨31⟩

  def union (a b : Rights) : Rights := ⟨a.bits ||| b.bits⟩
  def inter (a b : Rights) : Rights := ⟨a.bits &&& b.bits⟩
  def subset (a b : Rights) : Bool := (a.bits &&& b.bits) = a.bits

  instance : Lattice Rights where
    sup := union
    inf := inter
    le := fun a b => subset a b
end Rights

-- Theorem: Rights form a bounded lattice
theorem rights_lattice : BoundedLattice Rights := by
  constructor
  · exact Rights.all   -- top
  · exact Rights.none  -- bottom
  · intro a; exact Rights.subset_none a
  · intro a; exact Rights.subset_all a
```

### 5.3 Derivation Rules

Capability delegation follows strict rules:

```lean
-- REQ-CAP-003: Delegation cannot amplify
def derive (parent : Capability R) (newRights : Rights)
           (actor : Actor) : Option (Capability R) :=
  if Rights.subset newRights parent.rights then
    some {
      resource   := parent.resource,
      rights     := newRights,
      provenance := parent.provenance.extend actor,
      epoch      := parent.epoch
    }
  else
    none  -- Cannot amplify rights

-- Fundamental: derived capabilities cannot exceed parent
-- REQ-SEC-003 verification
theorem no_amplification (parent child : Capability R) :
    child.derivedFrom parent →
    Rights.subset child.rights parent.rights := by
  intro h_derived
  exact derivation_preserves_bounds h_derived
```

---

## 6. Capability Model

### 6.1 Capability Structure

```lean
-- Complete capability definition
-- REQ-CAP-001: resource + rights + provenance + epoch
structure Capability (Resource : Type) where
  resource   : Resource        -- The protected resource
  rights     : Rights          -- Permitted operations
  provenance : Provenance      -- Derivation chain
  epoch      : Epoch           -- Validity epoch

structure Provenance where
  root      : CapabilityId     -- Original source
  chain     : List DelegationRecord

structure DelegationRecord where
  delegator : Actor
  delegatee : Actor
  timestamp : Timestamp
  constraints : Option Constraints
```

### 6.2 Three Security Properties

**These three properties are AETHEROS's security foundation:**

| Property | Statement | Verification |
|----------|-----------|--------------|
| **No Forgery** | Capabilities cannot be created without derivation from existing capability | Lean 4 proof |
| **No Amplification** | Derived capability cannot have more rights than parent | Lean 4 proof |
| **Revocation Propagates** | When parent is revoked, all descendants become invalid | TLA+ model |

```lean
-- Property 1: No Forgery
-- REQ-SEC-002
theorem no_forgery (cap : Capability R) (s : SystemState) :
    cap ∈ s.validCapabilities →
    cap.provenance.root ∈ s.governance.capabilityRoots := by
  intro h_valid
  exact governance_tracks_all_roots cap s h_valid

-- Property 2: No Amplification
-- REQ-SEC-003
theorem no_amplification (parent child : Capability R) :
    child.derivedFrom parent →
    Rights.subset child.rights parent.rights := by
  intro h_derived
  cases h_derived with
  | direct h_derive =>
    exact derive_restricts parent child h_derive
  | transitive p h_p h_c =>
    have h1 := no_amplification parent p h_p
    have h2 := no_amplification p child h_c
    exact Rights.subset_trans h2 h1

-- Property 3: Revocation Propagates
-- REQ-SEC-004
theorem revocation_propagates (cap : Capability R) (s s' : SystemState) :
    s' = revokeCapability s cap →
    ∀ child : Capability R, child.derivedFrom cap →
      ¬ (child ∈ s'.validCapabilities) := by
  intro h_revoke child h_derived
  exact revocation_invalidates_descendants s s' cap child h_revoke h_derived
```

### 6.3 Capability Operations

```lean
-- Core capability operations

-- Create (only Governance kernel, for roots)
def createRootCapability (gov : GovernanceState) (resource : R)
                         (rights : Rights) (actor : Actor)
    : GovernanceState × Capability R :=
  let cap := {
    resource := resource,
    rights := rights,
    provenance := Provenance.root (gov.nextCapId),
    epoch := gov.currentEpoch
  }
  let gov' := { gov with
    nextCapId := gov.nextCapId + 1,
    capabilityRoots := gov.capabilityRoots.insert cap.id cap
  }
  (gov', cap)

-- Derive (any holder with GRANT right)
def deriveCapability (parent : Capability R) (newRights : Rights)
                     (delegator delegatee : Actor)
    : Option (Capability R) :=
  if ¬ parent.rights.hasGrant then none
  else if ¬ Rights.subset newRights parent.rights then none
  else some {
    resource := parent.resource,
    rights := newRights,
    provenance := parent.provenance.extend {
      delegator := delegator,
      delegatee := delegatee,
      timestamp := now(),
      constraints := none
    },
    epoch := parent.epoch
  }

-- Revoke (holder with REVOKE right, or Governance)
def revokeCapability (gov : GovernanceState) (cap : Capability R)
    : GovernanceState :=
  { gov with
    revokedCapabilities := gov.revokedCapabilities.insert cap.id,
    currentEpoch := gov.currentEpoch + 1
  }

-- Check validity
def isValid (gov : GovernanceState) (cap : Capability R) : Bool :=
  cap.epoch ≤ gov.currentEpoch ∧
  cap.provenance.root ∈ gov.capabilityRoots ∧
  ¬ (cap.id ∈ gov.revokedCapabilities) ∧
  ¬ (cap.provenance.chain.any (·.delegator ∈ gov.revokedActors))
```

---

## 7. Quadripartite Kernel Architecture

### 7.1 Governance Kernel (Nous/司)

**Role**: Constitutional authority, policy enforcement, capability management.

```lean
-- REQ-KERN-001: Governance enforces constitutional invariants
structure GovernanceState where
  capabilityStore   : CapabilityStore     -- All capability roots
  policyRules       : PolicyRuleSet       -- Active policies
  auditLog          : AuditLog            -- Immutable trail
  invariants        : InvariantSet        -- Constitutional laws
  actorRegistry     : ActorRegistry       -- Known actors
  arbitrationQueue  : Queue Dispute       -- Pending conflicts
  currentEpoch      : Epoch               -- System version

-- Constitutional invariants (cannot be disabled)
structure ConstitutionalInvariants where
  userPrimacy       : Invariant           -- Human authority supreme
  noForgery         : Invariant           -- Capabilities unforgeable
  noAmplification   : Invariant           -- Rights never increase
  revocationWorks   : Invariant           -- Revocation propagates
  auditComplete     : Invariant           -- All actions logged
```

**Operations:**

| Operation | Description | Capability Required |
|-----------|-------------|---------------------|
| `createCapability` | Create new capability root | GOVERNANCE |
| `revokeCapability` | Invalidate capability and descendants | REVOKE on target |
| `registerPolicy` | Add new policy rule | POLICY_ADMIN |
| `arbitrate` | Resolve capability conflict | GOVERNANCE |
| `queryAudit` | Read audit log | AUDIT_READ |

### 7.2 Physical Kernel (Soma/体)

**Role**: Hardware abstraction, memory management, interrupt handling, I/O.

```lean
-- REQ-KERN-002: Physical abstracts all hardware
structure PhysicalState where
  memoryDomains    : DomainMap            -- Isolated address spaces
  pageAllocator    : PageAllocator        -- Physical memory
  interruptTable   : InterruptTable       -- Handler dispatch
  deviceRegistry   : DeviceRegistry       -- Known devices
  timerState       : TimerState           -- Scheduling timers
  ioQueues         : Map DeviceId IOQueue -- Pending I/O

-- Memory domain with explicit coherence
structure MemoryDomain where
  id          : DomainId
  pages       : Set PageFrame
  coherence   : Coherence               -- coherent | explicit
  persistence : Bool                    -- survives power loss?
  owner       : Capability MemoryDomain

inductive Coherence where
  | coherent   -- Hardware-maintained (CPU)
  | explicit   -- Requires sync primitives (GPU)
```

**Interrupt Handling:**

```lean
-- Interrupt classification by latency requirement
inductive InterruptClass where
  | critical   : InterruptClass  -- NMI, MCE: <1μs
  | realtime   : InterruptClass  -- Timer, IPI: <10μs
  | deferred   : InterruptClass  -- Device I/O: <100μs

structure InterruptDescriptor where
  vector    : UInt8
  class     : InterruptClass
  handler   : InterruptHandler
  priority  : Priority
  affinity  : CPUSet              -- Which cores can handle
```

### 7.3 Emotive Kernel (Thymos/気)

**Role**: User intent inference, priority computation, experience quality.

```lean
-- REQ-KERN-003: Emotive maintains user intent model
structure EmotiveState where
  intentModel       : IntentModel         -- P(user wants X)
  presenceState     : PresenceState       -- User engagement level
  priorityCache     : PriorityCache       -- Computed priorities
  experienceHistory : ExperienceLog       -- Quality observations
  healthMonitor     : SystemHealth        -- Resource pressure
  lastUpdate        : Timestamp           -- Staleness tracking

-- Intent model as Bayesian inference
structure IntentModel where
  goals         : Map GoalId Probability  -- P(goal is active)
  evidence      : List Observation        -- Recent observations
  prior         : GoalDistribution        -- Background rates
  updateBudget  : Nat                     -- Cycles allowed

-- User presence detection
inductive PresenceLevel where
  | activeFocus     -- User actively engaged
  | passiveMonitor  -- User watching, not acting
  | background      -- User away, system visible
  | suspended       -- System can sleep
```

**Priority Computation:**

```lean
-- Experience-aware priority
def computePriority (intent : IntentModel) (task : Task) : Priority :=
  let goalRelevance := intent.goals.get task.associatedGoal |>.getD 0.0
  let urgency := task.deadline.map (fun d => 1.0 / (d - now())) |>.getD 0.0
  let userImpact := task.experienceImpact

  Priority.fromScore (
    0.4 * goalRelevance +
    0.3 * urgency +
    0.3 * userImpact
  )
```

### 7.4 Cognitive Kernel (Logos/理)

**Role**: Computation scheduling, GPU/NPU orchestration, dataflow execution.

```lean
-- REQ-KERN-004: Cognitive schedules all computation
structure CognitiveState where
  computeGraphs     : Map GraphId ComputeGraph
  scheduler         : SchedulerState
  placementPolicy   : PlacementPolicy
  resourceModel     : WorldModel          -- What's available
  pendingWork       : PriorityQueue Task
  runningTasks      : Map TaskId Execution

-- Dataflow computation graph
structure ComputeGraph where
  nodes     : List ComputeNode
  edges     : List DataEdge             -- Dependencies
  inputs    : List Port
  outputs   : List Port

structure ComputeNode where
  id        : NodeId
  kernel    : ComputeKernel             -- The actual computation
  target    : ComputeTarget             -- CPU/GPU/NPU preference
  estimate  : ResourceEstimate          -- Expected cost
```

**Scheduler Classes:**

| Class | Algorithm | Use Case |
|-------|-----------|----------|
| **Governance** | Run-to-completion | Policy decisions |
| **HardRealtime** | EDF (Earliest Deadline First) | Audio, control loops |
| **SoftRealtime** | EDF with slack | UI responsiveness |
| **Interactive** | CFS + Thymos hints | User-facing tasks |
| **Batch** | Fair-share | Background computation |
| **Idle** | Best-effort | Maintenance tasks |

**Core Partitioning (96 cores):**

```lean
def defaultPartition : CorePartition := {
  governanceCores   := CPUSet.range 0 2      -- Cores 0-1
  realtimeCores     := CPUSet.range 2 8      -- Cores 2-7
  interactiveCores  := CPUSet.range 8 32     -- Cores 8-31
  computeCores      := CPUSet.range 32 80    -- Cores 32-79
  systemCores       := CPUSet.range 80 96    -- Cores 80-95
}
```

---

## 8. System Call ABI

### 8.1 Register Convention (x86-64)

| Register | Purpose |
|----------|---------|
| RAX | Syscall number (input) / Return value (output) |
| RDI | Arg 1: **Capability token (always required)** |
| RDX | Arg 2: Operation-specific |
| RSI | Arg 3: Operation-specific |
| R10 | Arg 4: Operation-specific |
| R8 | Arg 5: Operation-specific |
| R9 | Arg 6: Operation-specific |

**Critical**: Every syscall requires a capability in RDI. No ambient authority.

### 8.2 Syscall Categories

| Category | Number Range | Target Kernel |
|----------|--------------|---------------|
| Capability | 0x0000-0x00FF | Governance |
| Memory | 0x0100-0x01FF | Physical |
| IPC | 0x0200-0x02FF | Physical (routed by Cognitive) |
| Process | 0x0300-0x03FF | Governance + Physical |
| Device | 0x0400-0x04FF | Physical |
| Compute | 0x0500-0x05FF | Cognitive |
| Intent | 0x0600-0x06FF | Emotive |

### 8.3 Error Handling

```rust
/// Unified syscall error type
#[repr(u64)]
pub enum SyscallError {
    // Capability errors (0x1xxx)
    CapabilityInvalid     = 0x1001,
    CapabilityRevoked     = 0x1002,
    CapabilityExpired     = 0x1003,
    InsufficientRights    = 0x1004,

    // Resource errors (0x2xxx)
    ResourceNotFound      = 0x2001,
    ResourceBusy          = 0x2002,
    ResourceExhausted     = 0x2003,

    // Operation errors (0x3xxx)
    InvalidOperation      = 0x3001,
    InvalidArgument       = 0x3002,
    WouldBlock            = 0x3003,
    Interrupted           = 0x3004,

    // Security errors (0x4xxx)
    AccessDenied          = 0x4001,
    PolicyViolation       = 0x4002,
    AuditFailure          = 0x4003,
}
```

---

## 9. Thread Model

### 9.1 Thread Structure

```lean
structure Thread where
  id            : ThreadId
  domain        : DomainId           -- Owning protection domain
  state         : ThreadState
  priority      : Priority
  cpuAffinity   : CPUSet
  stack         : StackDescriptor
  tls           : TLSDescriptor
  context       : RegisterContext
  pendingSignals: SignalQueue
  blockedOn     : Option BlockReason

structure StackDescriptor where
  base          : VirtAddr
  size          : Size
  guardPages    : Nat               -- Number of guard pages (default: 1)
  growable      : Bool

structure TLSDescriptor where
  base          : VirtAddr
  size          : Size
  initialized   : Bool
```

### 9.2 Thread States

```
                    ┌─────────────────────────────────────────────────┐
                    │                                                   │
                    ▼                                                   │
    ┌──────────┐ create ┌──────────┐ schedule ┌──────────┐            │
    │  Embryo  │───────►│  Ready   │─────────►│ Running  │────────────┤
    └──────────┘        └──────────┘          └──────────┘            │
                             ▲                     │                   │
                             │                     │ preempt/yield     │
                             └─────────────────────┘                   │
                             │                     │                   │
                        unblock                  block                 │
                             │                     │                   │
                             │                     ▼                   │
                        ┌──────────┐          ┌──────────┐            │
                        │ Blocked  │◄─────────│ Waiting  │            │
                        └──────────┘          └──────────┘            │
                                                   │                   │
                                                   │ terminate         │
                                                   ▼                   │
                                              ┌──────────┐            │
                                              │   Dead   │────────────┘
                                              └──────────┘       (reap)
```

### 9.3 Context Switch Path

```rust
/// Minimal context switch (hot path)
/// Target: < 1μs on Zen 5
#[naked]
unsafe extern "C" fn context_switch(
    old_ctx: *mut RegisterContext,
    new_ctx: *const RegisterContext
) {
    asm!(
        // Save callee-saved registers
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",

        // Save stack pointer
        "mov [rdi], rsp",

        // Load new stack pointer
        "mov rsp, [rsi]",

        // Restore callee-saved registers
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",

        "ret",
        options(noreturn)
    );
}
```

---

## 10. Memory Subsystem

### 10.1 Memory Domain Architecture

```lean
-- Memory isolation through domains
structure MemoryDomain where
  id          : DomainId
  pageTable   : PageTable           -- MMU mapping
  pages       : Set PageFrame       -- Physical memory
  coherence   : Coherence
  persistence : Bool
  quota       : MemoryQuota
  usage       : MemoryUsage
  owner       : Capability MemoryDomain

-- Page frame descriptor
structure PageFrame where
  pfn         : PhysAddr            -- Physical frame number
  order       : Nat                 -- 2^order pages (buddy allocator)
  flags       : PageFlags
  refCount    : Nat
  domain      : Option DomainId     -- Owner domain

-- Page flags
structure PageFlags where
  present     : Bool
  writable    : Bool
  executable  : Bool
  user        : Bool
  global      : Bool
  dirty       : Bool
  accessed    : Bool
  cacheMode   : CacheMode
```

### 10.2 Buddy Allocator

```rust
/// Buddy allocator for physical memory
/// O(1) allocation, O(log n) coalescing
pub struct BuddyAllocator {
    free_lists: [List<PageFrame>; MAX_ORDER + 1],
    total_pages: usize,
    free_pages: AtomicUsize,
}

impl BuddyAllocator {
    /// Allocate 2^order contiguous pages
    pub fn alloc(&mut self, order: usize) -> Option<PageFrame> {
        // Find smallest sufficient order
        for o in order..=MAX_ORDER {
            if let Some(page) = self.free_lists[o].pop() {
                // Split if needed
                while o > order {
                    o -= 1;
                    let buddy = page.buddy_at(o);
                    self.free_lists[o].push(buddy);
                }
                return Some(page);
            }
        }
        None
    }

    /// Free pages, coalescing with buddy if possible
    pub fn free(&mut self, page: PageFrame) {
        let mut order = page.order;
        let mut frame = page;

        while order < MAX_ORDER {
            let buddy = frame.buddy_at(order);
            if !self.try_remove(&buddy, order) {
                break;
            }
            frame = frame.merge(buddy);
            order += 1;
        }

        self.free_lists[order].push(frame);
    }
}
```

### 10.3 NUMA Awareness

```lean
-- NUMA topology for Threadripper PRO
structure NumaTopology where
  nodes       : Array NumaNode
  distances   : Matrix Nat          -- Hop counts between nodes

structure NumaNode where
  id          : NodeId
  cores       : CPUSet
  memory      : MemoryRange         -- Physical address range
  bandwidth   : Bandwidth           -- GB/s
  latency     : Latency             -- Nanoseconds

-- Allocation policy
inductive NumaPolicy where
  | local     : NumaPolicy          -- Allocate from local node
  | interleave: NumaPolicy          -- Round-robin across nodes
  | preferred : NodeId → NumaPolicy -- Prefer specific node
  | bind      : NodeId → NumaPolicy -- Strict binding
```

### 10.4 GPU/NPU Memory

```lean
-- Heterogeneous memory domains
structure GPUMemoryDomain extends MemoryDomain where
  vramBase    : PhysAddr
  vramSize    : Size
  barMapping  : Option VirtAddr     -- CPU-visible BAR

-- Memory coherence model
inductive GPUCoherence where
  | deviceOnly    -- Only GPU can access
  | hostVisible   -- CPU can read/write (uncached)
  | hostCached    -- CPU cached (explicit sync required)
  | managed       -- Runtime manages migration

-- Transfer operations
def gpuMemcpy (src dst : Pointer) (size : Size)
              (kind : TransferKind) : IO Unit := do
  match kind with
  | .hostToDevice => dmaTransfer src.toPhys dst.toDevice size
  | .deviceToHost => dmaTransfer src.toDevice dst.toPhys size
  | .deviceToDevice => gpuBlit src dst size
```

---

## 11. Inter-Kernel Communication

### 11.1 Typed Channels

```lean
-- Session-typed channels
-- REQ-IPC-001: Channels are session-typed
inductive SessionType where
  | send    : Type → SessionType → SessionType    -- !T.S
  | recv    : Type → SessionType → SessionType    -- ?T.S
  | choice  : SessionType → SessionType → SessionType  -- S₁ ⊕ S₂
  | offer   : SessionType → SessionType → SessionType  -- S₁ & S₂
  | rec     : (SessionType → SessionType) → SessionType  -- μX.S
  | end     : SessionType                         -- End

-- Channel endpoint
structure Endpoint (S : SessionType) where
  id        : ChannelId
  buffer    : RingBuffer Message
  partner   : Option EndpointId

-- Type-safe send
def send {A : Type} {S : SessionType}
    (ep : Endpoint (.send A S)) (msg : A)
    : IO (Endpoint S) := do
  ep.buffer.enqueue (encode msg)
  pure { ep with /* type changes to S */ }
```

### 11.2 Kernel Communication Protocols

```
┌─────────────┐                    ┌─────────────┐
│  GOVERNANCE │                    │  PHYSICAL   │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       │ ◄──── policyQuery ──────────────►│
       │                                  │
       │ ◄──── capabilityRequest ────────►│
       │                                  │
       │ ────► auditEvent ────────────────│
       │                                  │
┌──────┴──────┐                    ┌──────┴──────┐
│   EMOTIVE   │                    │  COGNITIVE  │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       │ ◄──── priorityHint ─────────────►│
       │                                  │
       │ ◄──── resourcePressure ─────────►│
       │                                  │
       │ ◄──── experienceGoal ───────────►│
```

### 11.3 Message Format

```rust
/// Universal kernel message format
#[repr(C)]
pub struct KernelMessage {
    header: MessageHeader,
    payload: [u8; MAX_PAYLOAD],
}

#[repr(C)]
pub struct MessageHeader {
    msg_type: MessageType,      // 2 bytes
    flags: MessageFlags,        // 2 bytes
    source_kernel: KernelId,    // 1 byte
    dest_kernel: KernelId,      // 1 byte
    sequence: u16,              // 2 bytes
    payload_len: u32,           // 4 bytes
    timestamp: u64,             // 8 bytes
    capability: CapabilityToken,// 16 bytes
}
```

### 11.4 IPC Latency Budget

| Path | Target | Notes |
|------|--------|-------|
| Same-core | < 500 ns | Cache-hot, no TLB flush |
| Cross-core, same CCX | < 800 ns | L3 shared |
| Cross-CCX | < 1.2 μs | CCD-to-CCD |
| Cross-CCD | < 1.5 μs | Infinity Fabric |

---

## 12. Security Threat Model

### 12.1 Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRUST BOUNDARY DIAGRAM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─ RING -1 (HYPERVISOR) ───────────────────────────────────────────────┐  │
│  │  AMD-V / SEV-SNP: Hardware root of trust                             │  │
│  │  [Threat: Hardware backdoors, side channels]                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─ RING 0 (MICROKERNEL) ──────────▼────────────────────────────────────┐  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │  GOVERNANCE  │  │   PHYSICAL   │  │  Capability  │               │  │
│  │  │    KERNEL    │◄─┤    KERNEL    ├─►│   Routing    │               │  │
│  │  │  (authority) │  │  (hardware)  │  │  (dispatch)  │               │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  │  [Threats: Kernel bugs, privilege escalation, IPC manipulation]      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─ RING 3 (UNPRIVILEGED) ─────────▼────────────────────────────────────┐  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │  │   EMOTIVE    │  │  COGNITIVE   │  │    USER      │               │  │
│  │  │    KERNEL    │  │    KERNEL    │  │   DOMAINS    │               │  │
│  │  │  (advocacy)  │  │  (schedule)  │  │ (applications)│              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│  │  [Threats: Malicious apps, confused deputy, resource exhaustion]     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─ EXTERNAL ──────────────────────▼────────────────────────────────────┐  │
│  │  Network │ Storage │ Peripherals │ Side Channels                     │  │
│  │  [Threats: Network attacks, malicious devices, physical access]      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Threat Categories

| Category | Attack Vector | Mitigation |
|----------|---------------|------------|
| **Capability Forgery** | Memory corruption, type confusion | Cryptographic tokens, Lean 4 proofs |
| **Privilege Escalation** | Kernel bugs, IPC exploitation | Minimal TCB, formal verification |
| **Confused Deputy** | Service acting on malicious input | Capability passing, not ambient authority |
| **Resource Exhaustion** | Memory bombs, fork bombs | Quota enforcement, OOM handling |
| **Side Channels** | Timing, cache, speculative execution | Constant-time crypto, partition isolation |
| **Network Attacks** | Remote exploitation | Public channel isolation from System |

### 12.3 Capability-Based Defenses

```lean
-- Defense: Capability unforgeable
-- Attack: Memory corruption creates fake capability
-- Mitigation: Cryptographic MAC on capability tokens
structure SecureCapability where
  payload : CapabilityPayload
  mac     : HMAC                    -- Keyed hash

def verifyCapability (key : SymmetricKey) (cap : SecureCapability) : Bool :=
  hmac key cap.payload = cap.mac

-- Defense: No ambient authority
-- Attack: Confused deputy uses caller's privileges
-- Mitigation: Explicit capability passing
def protectedService (inputCap : Capability Resource)
                     (outputCap : Capability Result) : IO Result := do
  -- Service uses ONLY the capabilities explicitly passed
  -- Cannot access anything else
  let data ← read inputCap
  let result := process data
  write outputCap result
  pure result
```

### 12.4 Network Channel Isolation

```lean
-- REQ-SEC-006: System actor NEVER receives from public channels
-- Critical security invariant

inductive ChannelClass where
  | internal : ChannelClass    -- System telemetry (priority 1)
  | private  : ChannelClass    -- User-initiated (priority 2)
  | public   : ChannelClass    -- External internet (priority 3)

def canReceive (actor : Actor) (channel : ChannelClass) : Bool :=
  match actor, channel with
  | .system _, .public => false   -- CRITICAL: Never allowed
  | .system _, .internal => true
  | .system _, .private => false  -- Must be explicitly allowed
  | .human _, _ => true
  | .si _, .public => false       -- SI also restricted
  | .si _, _ => true

theorem system_never_receives_public :
    ∀ (s : Actor) (msg : Message),
      s.isSystem → msg.sourceChannel = .public →
      ¬ canDeliver msg s := by
  intro s msg h_system h_public
  simp [canReceive, h_system, h_public]
```

---

## 13. Language Assessment

### 13.1 The Critical Tension

Soft real-time demands predictability—bounded worst-case execution time (WCET). This creates friction with otherwise attractive language choices.

### 13.2 Language Comparison

| Criterion | C | Rust | Ada/SPARK | Assessment |
|---|---|---|---|---|
| *WCET Analyzability* | Mature tooling (aiT, RapiTime) | Tooling nascent | Decades of deployment | C and Ada lead |
| *Memory Safety* | Manual, error-prone | Compile-time guarantees | SPARK subset provable | Rust and SPARK lead |
| *Certification Paths* | DO-178C, IEC 61508 established | Ferrocene exists but thin | Mature ecosystem | C and Ada lead |
| *Timing Transparency* | No hidden costs | Bounds checking has costs | Ravenscar is deterministic | C and Ada lead |
| *Concurrency Safety* | Manual, race-prone | Compile-time race prevention | Protected objects | Rust leads |
| *Developer Ecosystem* | Massive, aging | Growing, enthusiastic | Niche | C leads size; Rust momentum |
| *AMD Driver Compatibility* | Native (ROCm is C/C++) | FFI required | FFI required | C leads |

### 13.3 C: The Incumbent

Why C dominates RTOS (QNX, VxWorks, RTEMS, Zephyr):
- WCET analyzability with mature tooling
- Certifiable via established standards
- No hidden costs—what you write is what executes
- Timing transparency—no GC pauses

*Weakness:* Memory safety requires heroic discipline.

### 13.4 Rust: The Modern Contender

Friction points for hard real-time:

| Concern | Issue |
|---|---|
| WCET analysis | Tooling nascent; bounds checking timing |
| Certification | Ferrocene exists but ecosystem thin |
| `async`/`await` | Non-deterministic timing |
| Panic handling | Runtime implications in `no_std` |

*Strength:* Compile-time memory safety. For 96 cores, transformative.

Rust _can_ work—see Oxide Computer's Hubris—but you build tooling alongside.

### 13.5 Ada/SPARK: The Underappreciated

Worth serious consideration:
- Ravenscar profile: deterministic tasking
- SPARK subset: formally provable
- Mature WCET analysis: aerospace/defense deployment
- Native concurrency: protected objects, rendezvous

*Weakness:* AMD ecosystem is C-centric. FFI adds complexity.

### 13.6 AETHEROS Language Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│  Timing-Critical Core (if hard real-time needed)               │
│  ───────────────────────────────────────────────                │
│  C (scheduler hot paths) + Ada/SPARK (safety-critical)         │
│  Assembly (interrupt handlers, context switch)                  │
├─────────────────────────────────────────────────────────────────┤
│  Primary System Implementation                                  │
│  ─────────────────────────────                                  │
│  Rust (kernels, IPC, memory management, drivers)               │
│  Compile-time safety justifies minor timing overhead            │
├─────────────────────────────────────────────────────────────────┤
│  Heterogeneous Compute Integration                              │
│  ─────────────────────────────────                              │
│  C++ (ROCm/HIP integration, existing AMD tooling)              │
│  Custom DSL (GPU/NPU compute kernels)                          │
└─────────────────────────────────────────────────────────────────┘
```

**Decision:** *Rust is the primary choice* with strategic C/C++ for AMD integration.

---

## 14. Heterogeneous Compute

### 14.1 GPUs Are Fundamentally Non-Deterministic

AMD's RDNA/CDNA architectures exhibit:

| Characteristic | Timing Implication |
|---|---|
| *Variable warp scheduling* | Execution order non-deterministic |
| *Memory coalescing variance* | Latency unpredictable |
| *Driver-level preemption* | Context switches imprecise |
| *Thermal throttling* | Performance varies with temperature |
| *Resource contention* | Multiple kernels create interference |

### 14.2 Implications for Soft Real-Time

| Strategy | Description |
|---|---|
| *Isolation* | GPU never blocks latency-critical ops |
| *Conservative WCET* | Budget 2-3x typical execution |
| *Statistical guarantees* | Soft deadlines, not hard |
| *Preemption points* | Known-duration segments |
| *Fallback paths* | CPU can complete if GPU misses deadline |

### 14.3 The NPU Difference

AMD's XDNA is more predictable:
- Fixed-function tiles (less scheduling variance)
- DMA-based data movement (predictable transfers)
- Inference workloads (consistent iteration counts)

The Emotive kernel may leverage NPU for intent inference with tighter timing.

---

## 15. GPU/NPU DSL Design

> **Note**: This describes a *future direction*. MVK uses HIP/ROCm directly.

### 15.1 Current Options Are Inadequate

| Option | Limitation |
|---|---|
| *CUDA/HIP* | Low-level, AMD-specific |
| *OpenCL* | Effectively abandoned |
| *WGSL* | Web-constrained |
| *MLIR/LLVM* | Right abstraction, wrong ergonomics |
| *Triton* | Python dependency, JIT overhead |

### 15.2 Algorithm/Schedule Separation

Inspired by Halide: *separate what to compute from how to compute it*.

```
Traditional: Algorithm + Schedule + Target intertwined

AETHEROS:
┌─────────────────────────────────────────┐
│  Algorithm (pure, functional)           │
│  "What to compute"                      │
├─────────────────────────────────────────┤
│  Schedule (optimization strategy)       │
│  "How to parallelize, tile, fuse"       │
├─────────────────────────────────────────┤
│  Target (hardware binding)              │
│  "CPU / GPU / NPU"                      │
└─────────────────────────────────────────┘
```

### 15.3 Proposed DSL

```lean
-- Functional, data-parallel core
def matmul (A : Tensor [M, K]) (B : Tensor [K, N]) : Tensor [M, N] :=
  Tensor.generate (M, N) fun (i, j) =>
    Σ k : Fin K, A[i, k] * B[k, j]

-- Schedule is separate
schedule matmul with
  | tile (i, j) by (32, 32)
  | parallelize i
  | vectorize j
  | target GPU.workgroup

-- Compiles to: SPIR-V, GCN/RDNA ISA, XDNA binary
```

### 15.4 Compilation Targets

| Target | Format | Use Case |
|---|---|---|
| *Portable* | SPIR-V | Cross-vendor |
| *AMD GPU* | GCN/RDNA ISA | Maximum performance |
| *AMD NPU* | XDNA binary | Inference acceleration |
| *CPU Fallback* | Native LLVM | Development, compatibility |

### 15.5 MVK Approach

**Deferred to post-MVK.** For MVK:
1. Use HIP/ROCm directly
2. Wrap in Rust safety abstractions
3. Collect data to inform DSL design

---

## 16. AI-Assisted Development Model

### 16.1 Claude Code Capabilities

| Capability | Assessment |
|---|---|
| *Generate Rust/Assembly* | Strong, with human review |
| *Architectural Consistency* | Good within context (~200K tokens) |
| *Implement Known Algorithms* | Strong |
| *Translate Specs to Code* | Reasonable if specs precise |
| *Iterate on Feedback* | Excellent |
| *Generate TLA+/Lean 4* | Moderate to strong |

### 16.2 Claude Code Limitations

| Limitation | Implication |
|---|---|
| *No Persistent State* | Human maintains memory across sessions |
| *Cannot Test on Hardware* | Human validates physically |
| *Cannot Execute OS* | Debugging requires human observation |
| *Context Bounds* | Cannot hold entire OS simultaneously |
| *Subtle Errors Possible* | Low-level code, memory ordering |
| *No Accountability* | Human must verify all output |

### 16.3 Collaboration Topology

```
┌─────────────────────────────────────────────────────────┐
│                        HUMAN                            │
│   • Architectural authority                             │
│   • Mathematical foundations                            │
│   • Hardware validation                                 │
│   • Final correctness responsibility                    │
├─────────────────────────────────────────────────────────┤
│                    CLAUDE CODE                          │
│   • Implementation at human direction                   │
│   • Translation of specs to code                        │
│   • Refactoring toward simplicity                       │
│   • Documentation and explanation                       │
│   • Tedious work (test generation, etc.)                │
├─────────────────────────────────────────────────────────┤
│                  FORMAL TOOLS                           │
│   • Proof assistants (Lean 4, Coq)                      │
│   • Model checkers (TLA+, SPIN)                         │
│   • The arbiter of mathematical truth                   │
└─────────────────────────────────────────────────────────┘
```

### 16.4 Recommended Workflow

1. Human defines mathematical intent
2. Claude drafts formal specifications
3. Human verifies with tools
4. Iterate until proven
5. Claude generates implementation
6. Human validates on hardware

### 16.5 Force Multiplier Estimate

| Phase | Without AI | With AI |
|---|---|---|
| Specification | 1x | 0.3-0.5x |
| Implementation | 1x | 0.5-0.7x |
| Debugging | 1x | 0.8-1.0x |
| Hardware integration | 1x | 0.9-1.0x |
| Documentation | 1x | 0.2-0.3x |

**Total compression: 30-40%**—meaningful but not an order of magnitude.

---

## 17. Existing Work to Study

### 17.1 Systems and Insights

| System | Key Insight |
|---|---|
| *seL4* | Capability model done right. Formal verification of C. Gold standard. |
| *Redox OS* | Rust microkernel viable. Unix-like familiarity. |
| *Fuchsia/Zircon* | Object-capability at scale. Handles, not file descriptors. |
| *Hubris* | Rust RTOS for production. Safety-critical viability. |
| *Theseus* | Rust OS with live evolution. Type-enforced invariants. |
| *Legion* | Dataflow for heterogeneous compute. Realm runtime. |
| *Halide* | Algorithm/schedule separation. |

### 17.2 Foundational Research

| Area | Key Works |
|---|---|
| *Capability Systems* | Dennis & Van Horn (1966), EROS, seL4 |
| *Microkernel Design* | L4 family, seL4 proofs |
| *Formal Verification* | CompCert, CertiKOS, TLA+ |
| *Session Types* | Honda (1993), Gay & Hole |
| *Linear Types* | Girard, Wadler, Rust ownership |

### 17.3 Absorb vs. Invent

| Absorb (proven) | Adapt | Invent (novel) |
|---|---|---|
| Capability model (seL4) | Microkernel → quadripartite | Emotive kernel concept |
| Session types | Dataflow → user-centric | User intent inference |
| Memory safety (Rust) | GPU integration → native | Algorithm/schedule DSL |
| Formal methods | Persistence → checkpoint | Experience quality |

---

## 18. Complete System Walkthrough

### 18.1 Scenario: Document Open

User clicks document icon. All four kernels coordinate.

```
═══════════════════════════════════════════════════════════════════════════════
SCENARIO: User opens a document
═══════════════════════════════════════════════════════════════════════════════

TIME     KERNEL       EVENT                        STATE CHANGE
───────────────────────────────────────────────────────────────────────────────
t=0      Physical     Mouse click detected         interrupt → event queue
         Physical     → Emotive: sensorEvent(mouseClick, coords, t=0)

t=1μs    Emotive      Receive sensorEvent
         Emotive      Update intent: P(editing) = 0.3 → 0.5
         Emotive      Presence: activeFocus
         Emotive      → Cognitive: attendTo(documentEditing, priority=high)

t=3μs    Cognitive    Receive attendTo
         Cognitive    Lookup application
         Cognitive    Schedule: loadDocument(docId)
         Cognitive    → Physical: scheduleCompute(loadDocGraph)

t=5μs    Physical     Allocate memory domain
         Physical     Issue NVMe read
         Physical     → Governance: requestCapability(docFile, read)

t=7μs    Governance   Check user has file capability: YES
         Governance   Derive restricted capability
         Governance   Log to audit
         Governance   → Physical: grantCapability(docFileRead)

t=9μs    Physical     Complete NVMe read
         Physical     → Cognitive: computeComplete(loadDoc, success)

t=11μs   Cognitive    Schedule: renderDocument on GPU
         Cognitive    → Physical: scheduleCompute(renderGraph, GPU)

t=500μs  Physical     GPU render complete
         Physical     → Cognitive: computeComplete(render, success)

t=504μs  Emotive      Experience: 504μs < 16ms ✓
         Emotive      P(editing) = 0.7
───────────────────────────────────────────────────────────────────────────────
SUMMARY
───────────────────────────────────────────────────────────────────────────────
  TOTAL LATENCY:          504μs (well within 16ms target)
  KERNEL HOPS:            12 messages
  CAPABILITIES EXERCISED: sensorEvent, docFileRead
  INVARIANTS MAINTAINED:  All capabilities from user authority
═══════════════════════════════════════════════════════════════════════════════
```

### 18.2 Invariants Verified

| Invariant | Status |
|---|---|
| Capability Derivation | ✓ PASS |
| No Forgery | ✓ PASS |
| Kernel Isolation | ✓ PASS |
| Latency SLO | ✓ PASS |
| Audit Completeness | ✓ PASS |

---

## 19. Development Phases

### 19.1 Minimum Viable Kernel (MVK)

| Component | MVK Scope | Full Scope |
|---|---|---|
| *Governance* | Capability CRUD, policy | + Audit, dynamic policy, arbitration |
| *Physical* | Memory, timer, serial, interrupts | + Full GPU/NPU drivers, NUMA |
| *Emotive* | Explicit preferences only | + Bayesian inference, presence |
| *Cognitive* | FIFO scheduler, CPU only | + Heterogeneous placement, EDF |

### 19.2 Implementation Roadmap

| Phase | Duration | Deliverable |
|---|---|---|
| **Foundation** | 3 months | Boot, memory, serial, capability primitives |
| **IPC** | 2 months | Typed channels, session types |
| **MVK** | 4 months | All four kernels running |
| **GPU Integration** | 3 months | Basic RDNA compute |
| **Userspace** | 3 months | Shell, file system, basic apps |
| **Optimization** | 3 months | Meet performance targets |

---

## 20. Formal Verification Suite

### 20.1 TLA+ Specification

```tla
---------------------------- MODULE AETHEROS ----------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Resources,      \* Set of all resources
    Actors,         \* Set of all actors
    Rights,         \* Set of all rights
    MaxCapabilities \* Maximum capabilities in system

VARIABLES
    capabilities,   \* Set of valid capabilities
    revoked,        \* Set of revoked capability IDs
    epoch           \* Current system epoch

TypeInvariant ==
    /\ capabilities \subseteq [resource: Resources, rights: SUBSET Rights,
                               epoch: Nat, id: Nat]
    /\ revoked \subseteq Nat
    /\ epoch \in Nat

\* Safety: No capability can exceed its parent's rights
NoAmplification ==
    \A c \in capabilities:
        c.parent # NULL =>
            c.rights \subseteq c.parent.rights

\* Safety: Revocation propagates
RevocationPropagates ==
    \A c \in capabilities:
        c.parent.id \in revoked => c.id \in revoked

\* Liveness: Valid requests eventually complete
ProgressGuarantee ==
    \A req \in validRequests:
        <>(req \in completedRequests)
=========================================================================
```

### 20.2 Lean 4 Proofs

```lean
-- Core capability safety proofs

-- Theorem: Capability derivation preserves bounds
theorem derivation_preserves_bounds
    (parent child : Capability R)
    (h : child.derivedFrom parent) :
    child.rights ⊆ parent.rights := by
  induction h with
  | direct h_derive =>
    exact derive_restricts parent child h_derive
  | transitive p h_p h_c ih_p ih_c =>
    calc child.rights
        ⊆ p.rights := ih_c
      _ ⊆ parent.rights := ih_p

-- Theorem: System preserves user primacy
theorem user_primacy_preserved
    (s s' : SystemState)
    (t : Transition)
    (h_valid : t.isValid s)
    (h_effect : s' = t.effect s)
    (h_primacy : s.satisfiesUserPrimacy) :
    s'.satisfiesUserPrimacy := by
  apply governance_enforces_primacy
  exact ⟨h_valid, h_effect, h_primacy⟩
```

---

## 21. POSIX Compatibility

### 21.1 Compatibility Layer

AETHEROS provides a thin POSIX shim for application portability:

| POSIX Function | AETHEROS Mapping |
|----------------|------------------|
| `open()` | Capability request + channel creation |
| `read()`/`write()` | Channel send/receive |
| `mmap()` | Memory domain creation |
| `pthread_*` | Domain thread creation |

### 21.2 Not Supported

| Function | Reason |
|----------|--------|
| `fork()` | Capability model incompatible with COW semantics |
| `setuid()` | Ambient authority; capabilities instead |
| `signals` | Exception channels instead |

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| *Actor* | Principal with identity (Human, System, SI) |
| *Capability* | Unforgeable token encoding authority |
| *Channel* | Typed, capability-protected IPC conduit |
| *Domain* | Isolated state container |
| *Epoch* | Monotonic system version counter |
| *MA (間)* | Meaningful negative space principle |
| *Provenance* | Capability derivation chain |
| *Session Type* | Type encoding valid channel operation sequences |
| *SI* | Synthetic Intelligence actor |
| *Transition* | Pure function mapping state to state |

## Appendix B: Platform Specifications

| Component | Specification |
|-----------|---------------|
| CPU | AMD Threadripper PRO 9995WX |
| Cores | 96 cores / 192 threads |
| Architecture | Zen 5 |
| L3 Cache | 384 MB |
| Memory Channels | 12 DDR5 |
| PCIe | 5.0 (128 lanes) |
| TDP | 350W (configurable) |

## Appendix C: Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-01-01 | Architect | Initial specification |
| 1.0.0-S | 2025-01-15 | Architect | S-Tier enhanced version |
| 2.0.0 | 2025-12-26 | Consolidation | MA-compliant unified specification |

---

*This specification is the single source of truth for AETHEROS architecture and design.*
