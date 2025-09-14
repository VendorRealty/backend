# Floorplan Build Video with Gemini (Veo 3)

This folder contains a script that uses the Gemini API (Veo 3) to generate:

- a photorealistic house video from your floor plan, or
- a blueprint-style animation that draws walls, then electrical and plumbing overlays.

Input image: `nanobanana_animation/floorplan.png`
Output videos: `nanobanana_animation/output/*.mp4`

## Prerequisites

- Python 3.9+
- A Gemini API key with access to Veo 3 video generation.
  - If your account is not allowlisted for Veo 3, requests may fail with a permission error.
- The script will read `GEMINI_API_KEY` or `GOOGLE_API_KEY` from environment variables.
  - For convenience, it also reads `nanobanana_animation/.env` and will set `GEMINI_API_KEY` if it finds a line like `gemini_api_key=YOUR_KEY`.

## Install dependencies (recommended: virtual environment)

```bash
# Create and use a dedicated venv for this folder
python3 -m venv nanobanana_animation/.venv
nanobanana_animation/.venv/bin/python -m pip install --upgrade pip
nanobanana_animation/.venv/bin/python -m pip install -r nanobanana_animation/requirements.txt
```

## Run

```bash
# Photorealistic house transformation (default)
nanobanana_animation/.venv/bin/python nanobanana_animation/generate_floorplan_video.py --mode realistic

# Blueprint build animation (walls → electrical → plumbing)
nanobanana_animation/.venv/bin/python nanobanana_animation/generate_floorplan_video.py --mode blueprint
```

Outputs will be saved under `nanobanana_animation/output/`, e.g. `floorplan_realistic.mp4` or `floorplan_build.mp4`.

## Customization

Open `nanobanana_animation/generate_floorplan_video.py` and adjust:

- `prompt`: Tune instructions for the realistic walkthrough or blueprint animation.
- `aspect_ratio` in `GenerateVideosConfig` (e.g., `"1:1"`, `"16:9"`, `"9:16"`).
- You can also add parameters such as `duration_seconds`, `fps`, `negative_prompt`, etc., if supported by your Veo model version.

## Notes

- Model ID in the script is `veo-3.0-generate-001`. If you receive a permission error, try `veo-3.0-generate-preview` or ensure your key is enabled for Veo 3.
- Generation is asynchronous and can take a few minutes. The script polls until the job completes and then downloads the MP4.
