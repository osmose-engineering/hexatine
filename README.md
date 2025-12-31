# Glass Sponge Structures Visualizer (papers-integrated)

This mini-project is a **schematic / parametric** visualization toolkit for exploring structural motifs described in classic papers on hexactinellid (“glass”) sponges (especially *Euplectella aspergillum*).

It is **not** intended to be a photorealistic model. The goal is to make it easy to vary parameters and visually compare architectural ideas across length scales (macro lattice → micro spicules → nano silica).

## What’s new (based on the PDFs you uploaded)

The algorithm library (`glass_sponge_algorithms.py`) and the interactive notebook were updated to support motifs and parameter trends described in the papers, including:

- **Tapered body** (radius increases from base → top).
- Optional **two interpenetrating lattices** (a “two-grid proxy” for the quadrate lattice descriptions).
- Helical **ridges with**:
  - a **start height** (ridges absent in the narrow basal region),
  - a **pitch gradient** (e.g., tighter helix toward the top),
  - a **ridge-height gradient** (e.g., larger ribs toward the top),
  - ridges that follow the **tapered radius**.
- **Convex (domed) sieve plate** option (schematic).
- **Optical-waveguide spicule** tab: radial refractive-index profile + index map + derived NA/V-number.
- **Basalia spicule** tab: smooth + barbed region (schematic).
- Simple **indentation/fracture scaling** tab (normalized Pc ∝ Kc⁴/(E²·H)) for comparing regions.

## Files

- `glass_sponge_algorithms.py`  
  Core geometry generators and Plotly figure builders.

- `Glass_Sponge_Structures_Visualizer_papers_integrated.ipynb`  
  Jupyter notebook with a Gradio UI using the updated algorithms.

- `run_app.py`  
  Minimal script for rendering a default figure (optional; the notebook is the main UI).

## Quick start

1. Open `Glass_Sponge_Structures_Visualizer_papers_integrated.ipynb`
2. Run the cells top-to-bottom.
3. A Gradio UI should appear inline in the notebook.

If the UI doesn’t appear (Gradio/Jupyter quirks), re-run the final cell. The notebook calls:
`demo.launch(inline=True, prevent_thread_lock=True)` with fallbacks for different Gradio versions.

## Notes on units

Most macro/micro geometry uses **arbitrary units** for convenience and interactive speed.
The optical spicule tab uses **microns** (µm) because the paper reports refractive-index
features at that scale.

If you want a consistent physical unit system across the whole tool, add a global “scale factor”
and convert all parameters accordingly (the algorithms are written to make this straightforward).

