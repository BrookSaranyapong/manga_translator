# Type Error Fix Guide

## Fix Strategies by Error Type

### Broken import or moved file
→ Fix the import path to point to the correct module location.

### Type mismatch (`str` vs `int`, etc.)
→ Add an explicit cast or conversion (`int()`, `str()`, `cast()`, etc.)

### Third-party stub gap (`ultralytics`, `uharfbuzz`, `freetype`, etc.)
→ **Skip** — no fix needed. These are annotation bugs in external packages, safe to ignore.

### `isinstance()` with generic type
→ Use duck-typing (`hasattr` / `.item()`) instead of `isinstance()` with generic types.

## Verification

After making fixes, re-run `pyright` to confirm errors are resolved. Summarize:
- Total errors before
- How many fixed
- How many remaining (categorize as "needs fix" vs "external stub gap — safe to ignore")
