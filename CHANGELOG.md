# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and published versions will follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Established `vrp-gpu` as the public library identity and namespaced internal
  workspace packages.
- Added publication metadata, package-content checks and an explicit MSRV.
- Separated user-facing crate documentation from workspace contributor docs.

### Added

- A feature-gated CPU/GPU validation runner with raw latency samples, independent
  route-cost checks and documented public-API timing boundaries.

- Release, architecture and security documentation for future crates.io
  publication.

[Unreleased]: https://github.com/pedrozaz/vrp-gpu/compare/main...develop
