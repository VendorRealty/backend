Place your custom legend image here:

- File name: `electrical_legend.png`
- Recommended size: between 300x180 and 600x360 pixels
- Transparent background (optional) or white card
- Keep symbol colors consistent with overlays (yellow #FFC800)

The API `/api/legend/electrical` will serve this static file if present.
If missing, it falls back to a generated legend until you add your own.

You can also override the path via env:

- `ELECTRICAL_LEGEND_PATH=/absolute/path/to/your/electrical_legend.png`
