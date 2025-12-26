# AETHEROS - Claude Code Project Configuration

## Project Overview

**AETHEROS** is a research operating system implementing a quadripartite microkernel architecture.

## Architecture

| Kernel | Responsibility |
|--------|----------------|
| **Governance** | Security policy, capability management, constitutional enforcement |
| **Physical** | Hardware abstraction, drivers, CPU/GPU/NPU scheduling |
| **Emotive** | User intent inference, context modeling, experience optimization |
| **Cognitive** | Service orchestration, adaptation, self-healing behavior |

## Key Files

- `Specification.md` - Complete technical specification (120KB)
- `README.md` - Project overview and status
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - MIT License

## Development Status

| Phase | Status |
|-------|--------|
| Specification | Complete |
| Lean 4 Proofs | In Progress |
| MVK Prototype | Planned |
| Hardware Drivers | Planned |

## Core Principles

- **User Primacy Invariant**: System serves user as constitutional law
- **Capability-Based Security**: Fine-grained access control
- **MA Principle**: Minimal Artifact design philosophy
- **Formal Verification**: Lean 4 proofs for security-critical properties

## Working with This Project

When modifying the specification:
- Maintain consistent section numbering
- Cross-reference related sections
- Update the table of contents if structure changes
- Preserve Lean 4 proof syntax in formal verification sections

## Technologies

- **Proof Language**: Lean 4
- **Target Implementation**: Rust, C (on seL4 substrate)
- **Documentation**: Markdown with formal notation
