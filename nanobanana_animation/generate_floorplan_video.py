import os
import time
import argparse
from pathlib import Path
import mimetypes
import requests

from google import genai
from google.genai import types


def _load_api_key_from_env_file():
    """Load API key from a local .env if standard env vars are not set.

    The official client reads GEMINI_API_KEY or GOOGLE_API_KEY. If a local
    .env has a key like `gemini_api_key=...`, set GEMINI_API_KEY so the SDK
    picks it up automatically.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k.lower() in ("gemini_api_key", "google_api_key") and v:
                # Prefer GEMINI_API_KEY for the Developer API
                os.environ.setdefault("GEMINI_API_KEY", v)
                break
    except Exception:
        # Don't fail just because parsing .env failed; user may have proper env already
        pass


def main():
    _load_api_key_from_env_file()

    # Initialize client (will read GEMINI_API_KEY/GOOGLE_API_KEY from env)
    client = genai.Client()

    # CLI options
    parser = argparse.ArgumentParser(description="Generate Veo video from floorplan image")
    parser.add_argument(
        "--mode",
        choices=["realistic", "blueprint"],
        default="realistic",
        help="'realistic' to transform into a photorealistic 3D house walkthrough, or 'blueprint' to animate walls/electrical/plumbing",
    )
    parser.add_argument(
        "--gcs-output",
        help="Optional: When using Vertex AI, write outputs to this GCS prefix (e.g., gs://bucket/prefix)",
    )
    parser.add_argument(
        "--image-gcs-uri",
        help="Optional: When using Vertex AI, reference the input image from GCS instead of local file",
    )
    parser.add_argument(
        "--aspect",
        default=None,
        help="Override aspect ratio, e.g. '9:16', '16:9', or '1:1'",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Optional video resolution, e.g. '720p' or '1080p'",
    )
    parser.add_argument(
        "--negative-prompt",
        dest="negative_prompt",
        default=None,
        help="Optional negative prompt override",
    )
    # Model is fixed to Veo 3.0 (standard), with internal fallbacks to 3.0-fast and 3.0-preview
    args = parser.parse_args()

    here = Path(__file__).parent
    image_path = here / "floorplan.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing input image: {image_path}")

    if args.mode == "blueprint":
        # Prompt that animates build sequence: walls -> electrical -> plumbing
        prompt = (
            "Top-down 2D blueprint animation based on the provided reference floor plan image.\n"
            "Step 1: Start from a blank white sheet and animate the exterior and interior walls "
            "being drawn in bold black strokes that match the reference.\n"
            "Step 2: Overlay the electrical layout: animated yellow lines routed along interior "
            "walls, with small outlet symbols and light switches at doorways, all connecting "
            "back to a gray breaker panel near the garage.\n"
            "Step 3: Overlay the plumbing: blue lines for cold water, red lines for hot water, "
            "and thicker dark lines for drains/vents. Connect to fixtures in the bathrooms, "
            "kitchen, and laundry, merging at a main stack.\n"
            "Keep the camera static (no pan/zoom), crisp vector strokes, minimal shading, "
            "and a clean architectural plan style. Use the image strictly as the spatial "
            "reference so room shapes and positions match. Smooth time‑lapse build."
        )
        aspect = "1:1"
        negative_prompt = None
        out_filename = "floorplan_build.mp4"
    else:
        # Photorealistic transformation prompt
        prompt = (
            "Start by clearly showing the provided 2D architectural floor plan as a flat drawing on screen. "
            "Then transition into a photorealistic 3D reconstruction of the entire layout while preserving exact "
            "room sizes and positions. Extrude walls to full height and add floors, ceilings, and a roof that "
            "matches the footprint. Place doors and windows exactly where the plan specifies. Next, demonstrate "
            "the build sequence in animation: first, a timelapse of structural elements forming; then visualize the "
            "electrical system routing along interior walls with outlets and switches at doorways and lighting fixtures; "
            "after that, visualize the plumbing with blue lines for cold water, red lines for hot water, and thicker dark "
            "lines for drains/vents connecting to sinks, toilets, showers, laundry, and a main stack. Continue into a smooth, "
            "cinematic 3D walkthrough that showcases each room: living room, kitchen, bedrooms, bathrooms, office, laundry, "
            "closets, and garage. Progressively materialize realistic finishes (painted walls, flooring, cabinets, fixtures, "
            "lighting) with natural daylight and soft global illumination. Keep the layout perfectly consistent with the plan, "
            "avoid any on-screen text or watermarks, and maintain polished, professional visual quality."
        )
        aspect = "16:9"
        negative_prompt = "cartoon, drawing, low quality, watercolor, CGI-looking, over-saturated"
        out_filename = "floorplan_realistic.mp4"

    # Apply CLI overrides to match the user's example shape
    if args.aspect:
        aspect = args.aspect
    if args.negative_prompt is not None:
        negative_prompt = args.negative_prompt
    resolution = args.resolution

    # Detect if Vertex AI is enabled via env var
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")

    # Create an Image object for the starting frame/reference
    # If using Vertex and a GCS image is provided, prefer that; else fall back to inline bytes
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    if use_vertex and args.image_gcs_uri:
        image = types.Image(gcs_uri=args.image_gcs_uri, mime_type=mime)
    else:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image = types.Image(image_bytes=image_bytes, mime_type=mime)

    # Always use Veo 3.0
    model_id_primary = "veo-3.0-generate-001"
    fallbacks = ["veo-3.0-fast-generate-001", "veo-3.0-generate-preview"]

    # Build config dict to mirror the example usage
    cfg_kwargs = {"aspect_ratio": aspect}
    if resolution:
        cfg_kwargs["resolution"] = resolution
    if negative_prompt:
        cfg_kwargs["negative_prompt"] = negative_prompt
    config = types.GenerateVideosConfig(**cfg_kwargs)
    if use_vertex and args.gcs_output:
        # This field is supported by Vertex AI Veo APIs
        config.output_gcs_uri = args.gcs_output

    def try_generate(model_id: str, allow_strip_resolution: bool = True):
        nonlocal config
        try:
            return client.models.generate_videos(
                model=model_id,
                prompt=prompt,
                image=image,
                config=config,
            )
        except Exception as e:
            msg = str(e)
            # Retry without resolution if the model doesn't support it
            if allow_strip_resolution and "resolution" in msg.lower():
                if hasattr(config, "resolution") and config.resolution:
                    # Rebuild config without resolution
                    cfg_kwargs2 = {"aspect_ratio": aspect}
                    if negative_prompt:
                        cfg_kwargs2["negative_prompt"] = negative_prompt
                    config = types.GenerateVideosConfig(**cfg_kwargs2)
                    if use_vertex and args.gcs_output:
                        config.output_gcs_uri = args.gcs_output
                    # Retry once without resolution
                    return client.models.generate_videos(
                        model=model_id,
                        prompt=prompt,
                        image=image,
                        config=config,
                    )
            raise

    # Primary attempt
    try:
        operation = try_generate(model_id_primary)
    except Exception as e:
        # On certain errors, try fallbacks (only for veo3 variants)
        lowered = str(e).lower()
        retryable = any(k in lowered for k in [
            "permission", "unauthorized", "not enabled", "not found", "unknown model",
            "resource_exhausted", "quota", "rate limit", "failed_precondition"
        ])
        if fallbacks and retryable:
            for fb in fallbacks:
                try:
                    print(f"Retrying with fallback model '{fb}'…")
                    operation = try_generate(fb)
                    break
                except Exception as e2:
                    last_err = e2
            else:
                print(f"All fallbacks failed. Last error: {last_err}")
                raise
        else:
            print(f"Model '{model_id_primary}' failed: {e}")
            print("Tip: Ensure your key has quota for Veo video generation, or switch to Vertex AI with --gcs-output.")
            raise

    # Poll the long-running operation until it's done
    print("Submitted Veo job. Waiting for completion…")
    while not operation.done:
        time.sleep(10)
        operation = client.operations.get(operation)

    # Some SDK surfaces return `response`, others return `result`. Support both.
    response = getattr(operation, "response", None) or getattr(operation, "result", None)
    if not response or not getattr(response, "generated_videos", None):
        raise RuntimeError(f"Video generation failed or returned no videos: {operation}")

    generated = response.generated_videos[0]

    # Download or print GCS URI depending on API surface
    out_dir = here / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_filename

    video_obj = generated.video
    uri = getattr(video_obj, "uri", None)
    # Try SDK-managed download first
    try:
        client.files.download(file=video_obj)
        video_obj.save(str(out_path))
        print(f"Generated video saved to {out_path}")
        return
    except Exception:
        pass

    # If SDK download isn't available, try direct HTTP(S) download when possible
    if uri and isinstance(uri, str) and uri.startswith(("http://", "https://")):
        params = {}
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if "generativelanguage.googleapis.com" in uri and api_key:
            # Some endpoints require the API key as a query param
            params["key"] = api_key
        try:
            with requests.get(uri, params=params, stream=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"Generated video saved to {out_path}")
            return
        except Exception as e:
            print(f"Tried direct download but failed: {e}")

    # Fallback: inform user where the asset is
    if uri:
        print(f"Video available at: {uri}")
        print("If this is a gs:// URI, use 'gsutil cp' or rerun with Developer API to enable auto-download.")
    else:
        raise RuntimeError("No downloadable URI returned by the service.")


if __name__ == "__main__":
    main()
