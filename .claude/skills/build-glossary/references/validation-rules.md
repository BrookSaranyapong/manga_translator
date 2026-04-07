# Glossary Validation Rules

## Validation Checks

### Duplicates
- Check for duplicate `Chinese` entries (case-insensitive, whitespace-stripped)
- Fix: Remove duplicate rows (keep first, with the most complete `Note`)

### Empty Translations
- Flag rows where `Thai` is empty or missing while `Chinese` has a value

### Empty Keys
- Flag rows where `Chinese` is empty but `Thai` has a value

### Encoding
- Verify the file is saved as `utf-8-sig` (not plain `utf-8` or ANSI with mojibake)
- Fix: Re-save as `utf-8-sig` encoding

### Column Names
- Must be exactly: `Chinese`, `Thai`, `Note`

### Whitespace
- Strip whitespace from `Chinese` values

## Summary Output
- Total entries
- Duplicates removed
- Empty translations found
- Any translation rule violations
