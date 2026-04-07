# Target Module Architecture

## Desired Structure

```
modules/
├── detection/         yolo_detect_bubbles.py
├── ocr/               image_processor.py, debug_utils.py
├── translation/       translator.py, glossary_manager.py
├── rendering/         text_renderer.py
├── utils/             io.py (cv2 helpers + save_to_json)
└── pipeline/          orchestrator.py (MangaPipeline)
```

## Issues to Identify

1. **Tight coupling** — modules importing from each other in circular patterns
2. **God modules** — files with >1 class or >200 lines doing multiple responsibilities
3. **Flat namespace** — everything in a single `modules/` folder with no grouping
4. **Orphan imports** — files referenced by imports but moved/deleted

## Steps

1. Compare current structure against the target above
2. If files don't match, propose and implement moves with `git mv`
3. Update all imports to match new paths
4. Verify with `pyright` that no broken references were introduced
