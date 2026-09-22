# Default recipe lists available commands
default:
  @just --list

# Run fast workspace check
check:
  cargo check --workspace --all-targets

# Run check with all features enabled (including cudarc/gpu)
check-all:
  cargo check --workspace --all-targets --all-features

# Check formatting
fmt:
  cargo fmt --all --check

# Fix formatting automatically
fmt-fix:
  cargo fmt --all

# Run Clippy with warnings as errors
clippy:
  cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run workspace unit and integration tests
test:
  cargo test --workspace --all-features --locked

# Build public API documentation with the same features exposed on docs.rs
docs:
  RUSTDOCFLAGS="-D warnings" cargo doc -p vrp-gpu --all-features --no-deps --locked

# Assemble and verify the future crates.io archive without publishing it
package:
  cargo package -p vrp-gpu --allow-dirty --locked
  cargo package -p vrp-gpu --list --allow-dirty --locked | grep -Fxq LICENSE
  cargo package -p vrp-gpu --list --allow-dirty --locked | grep -Fxq README.md
  cargo package -p vrp-gpu --list --allow-dirty --locked | grep -Fxq kernel.ptx

# Run dependency and license check via cargo-deny
deny:
  cargo deny check

# Validate the standalone nightly kernel workspace
kernel-ci:
  #!/usr/bin/env sh
  set -eu
  cd crates/vrp-gpu-kernel
  cargo fmt --check
  cargo check --all-targets
  cargo clippy --all-targets -- -D warnings
  cargo test

# Full local CI suite (execute before committing or opening a PR)
ci: fmt check-all clippy test docs package deny
    @echo "✅ All local CI checks passed!"
