# run-pipeline

Run `python main.py`, monitor logs for errors (API quotas, OCR fails), and summarize results.

## Usage

1. Execute `python main.py` and capture all output
2. Check the output against [diagnostic rules](references/diagnostic-rules.md)
3. Report findings:
   - How many bubbles detected, how many had text, how many were translated
   - Any errors or warnings found
   - Whether the final output image was produced
