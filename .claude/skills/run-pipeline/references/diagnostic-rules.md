# Pipeline Diagnostic Rules

## Error Indicators to Scan

| Indicator | Meaning |
|---|---|
| `RESOURCE_EXHAUSTED` or `429` | Gemini API quota exceeded |
| `❌` | Missing files or fatal errors |
| `No text` on every bubble | OCR failed to read anything |
| `UNEXPECTED` | Model architecture warnings |
| `Exception` or `Traceback` | Python crashes |

## Reporting Checklist

- [ ] Number of bubbles detected
- [ ] Number that had text
- [ ] Number that were translated
- [ ] Any errors or warnings found
- [ ] Whether the final output image was produced
- [ ] If errors found: suggest most likely cause and fix
