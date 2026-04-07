# type-check

Run pyright on the project, report type errors, and fix broken references.

## Usage

1. Run `pyright .` (with or without `--outputjson`) on the project root
2. Parse the output and group errors by file
3. Apply fixes based on the [error fix guide](references/error-fix-guide.md)
4. Re-run pyright to confirm errors are resolved
5. Summarize: total errors before, how many fixed, how many remaining
