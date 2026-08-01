# checkpoints/

The trained MRI SR checkpoint goes here as `best_sr_model.pt`. The path is
hardcoded as `MRIModelBackend.WEIGHTS_PATH` in `tools/super_resolution.py`

If this file isn't present, `mri_model` backend selection in the UI still
works — it silently falls back to a Lanczos upscale (see
`tools/super_resolution.py`), and prints the exact reason to
stderr and shows it in the Streamlit sidebar's Diagnostics panel, so the
rest of the app remains fully testable without the weights and it's
obvious when you're looking at a fallback vs. real output.

Real-ESRGAN weights: drop `RealESRGAN_x4plus.pth` here too. Its path is
hardcoded as `RealESRGANBackend.WEIGHTS_PATH` in
`tools/super_resolution.py`
