# Contributing to AETHEROS

Thank you for your interest in contributing to AETHEROS! This document provides guidelines for contributing to the project.

## Ways to Contribute

AETHEROS welcomes contributions across multiple areas:

| Area | Skills | Current Priority |
|------|--------|------------------|
| **Formal Verification** | Lean 4, TLA+, Coq | High |
| **Microkernel Implementation** | Rust, C, low-level systems | Medium |
| **Driver Development** | GPU/NPU programming, CUDA, ROCm | Medium |
| **Documentation** | Technical writing, diagrams | Ongoing |
| **Security Analysis** | Threat modeling, penetration testing | High |

## Getting Started

1. **Read the Specification**: Familiarize yourself with [`Specification.md`](Specification.md)
2. **Understand the Architecture**: Study the quadripartite kernel model
3. **Check Existing Issues**: Look for issues labeled `good-first-issue` or `help-wanted`

## Development Process

### For Code Contributions

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/Aetheros.git
cd Aetheros

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes, then commit
git add .
git commit -m "feat: description of change"

# Push and create PR
git push origin feature/your-feature-name
```

### For Formal Verification (Lean 4)

```bash
# Install Lean 4 and Lake
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build the proofs
lake build

# Run verification
lake exe verify
```

### For Documentation

- Use Markdown format (`.md` files)
- Follow existing style in `Specification.md`
- Include diagrams where helpful (Mermaid format preferred)

## Commit Message Format

We use conventional commits:

```
<type>: <short description>

<optional body>

<optional footer>
```

**Types:**
- `feat`: New feature or capability
- `fix`: Bug fix
- `proof`: Formal verification addition/update
- `docs`: Documentation changes
- `refactor`: Code restructuring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat: add capability revocation syscall

proof: verify transitive closure property in Lean 4

docs: clarify scheduler preemption behavior
```

## Code Style

### Rust
- Use `rustfmt` with default settings
- Follow Rust API guidelines
- Document public interfaces with `///` comments

### C
- Kernel style (Linux kernel coding style variant)
- 8-character tabs for indentation
- No typedef for structs

### Lean 4
- Follow mathlib4 naming conventions
- Prefer structured proofs over tactics for critical paths
- Document theorem statements

## Pull Request Process

1. **Create Draft PR Early**: Open a draft PR to get early feedback
2. **Complete Checklist**:
   - [ ] Code compiles without warnings
   - [ ] Tests pass (if applicable)
   - [ ] Documentation updated
   - [ ] Commit messages follow convention
3. **Request Review**: Convert to ready and request review
4. **Address Feedback**: Respond to all comments
5. **Merge**: Maintainer will squash and merge

## Review Criteria

PRs are evaluated on:

- **Correctness**: Does it work as intended?
- **Security**: Does it maintain capability invariants?
- **Performance**: Is it efficient?
- **Clarity**: Is the code readable?
- **Consistency**: Does it match existing patterns?

## Formal Verification Requirements

For security-critical code, formal verification is required:

| Component | Required Proofs |
|-----------|----------------|
| Capability Operations | Monotonic authority, No privilege escalation |
| Scheduler | Bounded latency, Priority inversion prevention |
| Memory Management | Isolation guarantees, No use-after-free |
| IPC | Capability transfer correctness |

## Security Disclosure

If you discover a security vulnerability:

1. **Do NOT** open a public issue
2. Email security concerns to the maintainers privately
3. Allow 90 days for response before public disclosure

## Questions?

- Open a Discussion for general questions
- Open an Issue for specific bugs or features
- Review existing documentation first

## Code of Conduct

- Be respectful and inclusive
- Focus on technical merit
- Assume good faith
- Teach rather than criticize

---

*Thank you for helping build an operating system that truly serves users.*
