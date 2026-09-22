# Release and crates.io checklist

This project is not published yet. These steps prepare and verify a release;
they do not replace a maintainer's final review of the archive.

## 1. Stabilize the release

1. Create `release/vX.Y.Z` from an up-to-date `develop`.
2. Choose the version according to Semantic Versioning and update
   `workspace.package.version` in the root `Cargo.toml`.
3. Move relevant entries from `Unreleased` into a dated changelog section and
   update comparison links.
4. Confirm that README examples, feature descriptions and the MSRV are current.
5. Confirm that only `vrp-gpu` is publishable.

Crate names are allocated first-come, first-served. Recheck the intended name
immediately before the first publication:

```sh
cargo search vrp-gpu --limit 10
cargo info vrp-gpu
```

## 2. Validate the source tree

```sh
just ci
just kernel-ci
cargo test -p vrp-gpu --features gpu -- --ignored
```

The hardware-only command requires a supported NVIDIA GPU and driver. If kernel
source changed, regenerate PTX and verify the committed copy:

```sh
cd crates/vrp-gpu-kernel
cargo oxide inspect --arch sm_120
cmp vrp_gpu_kernel.ptx ../vrp-gpu/kernel.ptx
```

## 3. Inspect the package

Run from a clean worktree:

```sh
cargo package -p vrp-gpu --list
cargo package -p vrp-gpu --locked
```

Inspect `target/package/vrp-gpu-X.Y.Z.crate` and the normalized manifest under
`target/package/vrp-gpu-X.Y.Z/`. The archive must contain the source, crate
README and versioned `kernel.ptx`; it must not contain kernel compiler output,
workspace tools or credentials.

## 4. Publish deliberately

After the release pull request is approved and merged according to the branch
policy:

```sh
cargo publish -p vrp-gpu --locked --dry-run
cargo publish -p vrp-gpu --locked
```

Publishing is permanent: an uploaded version cannot be overwritten. Never put
a crates.io token in the repository, command history, logs or CI output.

## 5. Record the release

1. Create an annotated, signed `vX.Y.Z` tag for the published commit.
2. Push the tag and create matching GitHub release notes from the changelog.
3. Verify the crates.io page and docs.rs build, including the `gpu` feature API.
4. Merge the final release metadata back into `develop` if the Git workflow
   produced any release-only changes.
