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
  cargo fmt --check

# Fix formatting automatically
fmt-fix:
  cargo fmt

# Run Clippy with warnings as errors
clippy:
  cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run workspace unit and integration tests
test:
  cargo test --workspace --all-features

# Run dependency and license check via cargo-deny
deny:
  cargo deny check 

# Full local CI suite (execute before comitting or opening PR)
ci: fmt check-all clippy test deny 
    @echo "✅ All local CI checks passed!"
