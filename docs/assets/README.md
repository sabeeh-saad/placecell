# Documentation assets

## Mission architecture

[`mission-architecture.html`](mission-architecture.html) is the editable source for the
architecture image used in the main README and mission guide. It contains its own CSS
and uses local system fonts; no remote assets or web server are required.

Open the HTML in a browser to edit or inspect it. Export the PNG from the repository root:

```bash
python3 docs/assets/render_mission_architecture.py
```

The exporter requires a local Chrome or Chromium installation. Use `--browser` to select
a particular executable. It creates a temporary browser profile, measures the layout,
checks for overflowing text, and captures the entire diagram at twice its CSS resolution.
The output is [`mission-architecture.png`](mission-architecture.png).

Keep the HTML and generated PNG together when changing the diagram. The image describes
the experimental agent-enabled path for memory destinations; named places retain their
separate Nav2 completion semantics.

The existing [`architecture.svg`](architecture.svg) and [`architecture.png`](architecture.png)
show the underlying camera-to-memory and single-goal navigation pipeline.
