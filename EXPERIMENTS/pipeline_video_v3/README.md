# Explainer video v3: how the model is fed, and how the calibration works (~5 min)

Same pipeline as v2 (EXPERIMENTS/pipeline_video on branch `experimentation`): Kokoro af_sarah, manim 0.22, 1080p30.

    python extract3/v01_data.py          # .venv311, PYTHONPATH = repo root: data3/v01.npz + facts_v01.json
    python build3.py narration tts       # STORYBOARD3.md -> narration3/narration.json -> audio3/*.wav (video venv)
    python build3.py render concat       # renders3/EMG_torque_feeding_and_calibration.mp4

Data: EG session 1 at -0.038 rad (windows, split), YES session 2 at -0.381 rad seed 0 (calibration), saved N0 runs
(window counts, epochs). The learning curve (data3/y04.npz) is copied from v2: rebuilt copy of last semester model, HM.
