# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-29

### Added
- Initial public release of `redacteval`.
- Core evaluation API: `RedactionEvaluator` for redaction quality scoring.
- Matching and segmentation capabilities: 
    - Support entity alias mapping
    - strict/non-strict entity type matching
    - `coverage_threshold` for partial overlaps
    - configurable sentence segmentation (default regex + custom segmenter hook).
- Usability utilities: 
    - report formatting via `format_report`/`print_report` 
    - bundled demo data via `load_demo_data`.


