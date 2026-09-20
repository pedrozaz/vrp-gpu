# Contributing to VRP-GPU

Thank you for your interest in contributing to VRP-GPU. Please read this guide to understand our architecture, development workflow, and quality standards.

---

## 1. Development Environment

The project is structured with a strict separation between dev-time and publish-time dependencies:

- **General Development (`vrp-core`, `vrp-bench`, `vrp-cli`)**:
    - Requires only stable Rust (`rustup default stable`).
    - Does **not** require CUDA toolchains or nightly compilers.
- **Kernel Development (`vrp-kernel`)**:
    - Requires `cuda-oxide` and the pinned Rust nightly toolchain.
    - Requires `clang23` and `llvm23` installed on the host.
    - Uses a standalone workspace so root workspace checks remain stable-only.

---

## 2. Git Workflow and Branching

We follow a GitFlow-inspired workflow:

- `main`: Production and tagged releases only. Direct commits are forbidden.
- `develop`: Primary integration branch. All features merge here.
- `feature/<slug>`: Isolated units of work branching off `develop` (e.g., `feature/solomon-parser`).
- `release/<vX.Y.Z>`: Release stabilization branch.
- `hotfix/<slug>`: Critical fixes branched from `main`.

---

## 3. Commit Guidelines

- **Conventional Commits**: Format messages as `type(scope): imperative short description`.
    - Allowed types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.
- **Atomic Commits**: Each commit must be a coherent, compiling unit of work.
- **Signed Commits**: All commits must include Developer Certificate of Origin (`git commit -s`).
- **No Fast-Forward Merges**: Feature branches must be merged into `develop` using `--no-ff` to preserve history.

---

## 4. Quality Gate

Before opening a Pull Request or merging, ensure all checks pass:

```bash
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo test --workspace
```

Validate the dev-only kernel workspace separately:

```bash
cd crates/vrp-kernel
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
```

---
