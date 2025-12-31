
"""
run_app.py

Optional: run the Glass Sponge Visualizer as a local Gradio app
(outside Jupyter).

Usage:
    python run_app.py
"""

from glass_sponge_algorithms import (
    LatticeParams, RidgeParams, SievePlateParams, AnchorBundleParams,
    SpiculeLamellaParams, CompositeBeamParams, CruciformSpiculeParams,
    full_skeleton_figure, lattice_figure, spicule_cross_section_figure,
    composite_beam_cross_section_figure, anchor_bundle_figure,
    cruciform_spicule_figure,
)
import gradio as gr


def render_macro():
    lattice = LatticeParams()
    ridges = RidgeParams()
    sieve = SievePlateParams(z_at_top=lattice.height)
    anchor = AnchorBundleParams()
    return full_skeleton_figure(lattice, ridges, sieve, anchor)


with gr.Blocks() as demo:
    gr.Markdown("# Glass Sponge Structural Visualizer (schematic)")
    out = gr.Plot()
    demo.load(render_macro, inputs=None, outputs=out)

demo.launch()
