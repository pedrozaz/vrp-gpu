# Release and crates.io checklist

The first planned release is `0.1.0-alpha.1`. These steps prepare and verify a
release; they do not replace a maintainer's final review of the archive or
authorize an upload. Only the library package `vrp-gpu` is publishable.

## 1. Stabilize the release

1. Create `release/vX.Y.Z[-prerelease]` from an up-to-date `develop` after the
   readiness pull request has merged. The first target is
   `release/v0.1.0-alpha.1`.
2. Confirm the intended version in `workspace.package.version`, the internal
   workspace dependency version and `Cargo.lock` agree. Change all three if
   the release target changes.
3. Move relevant entries from `Unreleased` into a section dated on the actual
   release day. Add a link for the new tag and change the `Unreleased` link to
   compare that tag against `develop`.
4. Confirm that README examples, feature descriptions and the MSRV are current.
5. Confirm that only `vrp-gpu` is publishable.

Crate names are allocated first-come, first-served. Recheck the intended name
immediately before the first publication:

```sh
cargo search vrp-gpu --limit 10
(cd /tmp && cargo info vrp-gpu --registry crates-io)
```

Run `cargo info` outside this workspace: inside it, Cargo can resolve the
local package instead of the registry entry. Search results are not proof that
a name is free; the registry is authoritative when publishing. If the name is
already taken, stop and resolve the identity before changing release metadata.

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

Do not report a portable CI pass as a hardware test. If hardware validation
cannot be rerun for the release candidate, record the last tested tree and
the gap in the release review rather than claiming a fresh GPU pass.

## 3. Inspect the package

Run from a clean worktree:

```sh
cargo package -p vrp-gpu --list --locked
cargo package -p vrp-gpu --locked
cargo publish -p vrp-gpu --locked --dry-run
```

Inspect `target/package/vrp-gpu-X.Y.Z.crate` and the normalized manifest under
`target/package/vrp-gpu-X.Y.Z/`. The archive must contain the source, crate
README, `LICENSE` and versioned `kernel.ptx`; it must not contain kernel
compiler output, workspace tools or credentials. Use the actual version in the
archive path, including any `-alpha.N` suffix. The dry run does not publish.

## 4. Publish deliberately

After the release pull request is approved and merged according to the branch
policy, obtain explicit maintainer approval for the irreversible upload. For
the **first** release, `main` does not exist yet: create it from the reviewed
release commit and verify it points at exactly the source tree validated above.
For subsequent releases, merge the release branch into `main` through a PR.
Do not create a release tag for a tree different from the uploaded package.
Configure registry authentication through Cargo's supported login mechanism;
do not paste a token into a shell command or commit it. Then run:

```sh
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
