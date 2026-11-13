E02_minimal runs trained before the fix of 2026-10-05 13:30: validation loss
weights used the validation part's own per-angle variance instead of the
training part's (early-stopping criterion differed from the training loss).
Re-run in results/screen with the fixed code.  Kept only for traceability.
