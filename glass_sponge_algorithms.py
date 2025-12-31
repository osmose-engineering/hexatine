"""
glass_sponge_algorithms.py

Parametric (schematic) geometry generators and Plotly helpers for visualizing
key structural motifs reported in classic structural-biology / materials papers
on hexactinellid ("glass") sponges — especially *Euplectella aspergillum*.

This is intentionally **not** a photorealistic CAD model. Instead, it provides
compact algorithms that let you explore hierarchical architectural ideas
(nano → micro → macro) interactively.

What's new in this revision (integrated from the uploaded papers)
---------------------------------------------------------------
Macro / skeletal lattice
- **Tapered cylindrical lattice** (diameter increases from base → apex).
- Optional **two interpenetrating lattices** with an angular offset (a schematic
  proxy for the "two-grid" descriptions of the quadrate lattice).
- External **helical ridges** now support:
  - absent/starting partway up the body (ridges absent in narrow lower region),
  - **variable pitch** (e.g., decreasing pitch toward the apex),
  - **variable ridge height** (e.g., increasing ridge height toward the apex),
  - ridges follow the **tapered radius**.

Micro / spicules
- Laminated spicule model keeps the **graded silica-layer thickness** option.
  Defaults were adjusted so organic interlayers can be made much thinner
  (nm-scale vs micron-scale silica, if you choose dimensional scales).

Nano
- Nanoparticle rings can optionally scale marker size with radius (inner→outer)
  and can add an illustrative "subparticle" cloud around each nanoparticle.

Optics
- Adds an **optical-waveguide spicule** model with a radial refractive-index
  profile (high-index core, lower-index central cylinder, outer shell with
  weak oscillations), plus basic derived metrics (NA, acceptance angle, V-number).

Holdfast / basalia
- Adds a schematic **barbed basalia spicule** generator (smooth + barbed region).

The paper-derived numerical values are *not* forced as hard constraints; they
are available via presets/typical defaults in the UI notebook.

Dependencies
------------
- numpy
- plotly

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Dict, Any
import math
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:  # pragma: no cover
    raise ImportError("Plotly is required for this module. Try: pip install plotly") from e


# -----------------------------
# Type aliases
# -----------------------------

DiagonalMode = Literal["none", "all_cells", "checkerboard"]
RidgeHandedness = Literal["left", "right", "both"]


# -----------------------------
# Data containers
# -----------------------------

@dataclass(frozen=True)
class LatticeParams:
    """Parameters for an Euplectella-like cylindrical square lattice."""
    radius: float = 1.0              # base radius
    radius_top: Optional[float] = None  # if set: linear taper from base -> top
    height: float = 5.0
    n_theta: int = 24                # number of vertical struts around circumference
    n_z: int = 30                    # number of horizontal rings along height
    diagonal_mode: DiagonalMode = "checkerboard"
    diagonal_pair_offset: float = 0.02  # if >0, draw paired diagonals with tiny offset (purely visual)
    seam: bool = True                # close the cylinder (wrap-around)

    # Schematic proxy for "two interpenetrating square lattices" described in some analyses:
    interpenetrating: bool = False
    secondary_theta_offset: float = 0.5   # as a fraction of theta step (0.5 = half-cell)
    secondary_radial_offset: float = 0.0  # offset the secondary lattice radially for visibility

    def radius_at(self, z: float) -> float:
        """Radius at axial coordinate z, assuming linear taper if radius_top is set."""
        R0 = float(self.radius)
        if self.radius_top is None:
            return R0
        Rt = float(self.radius_top)
        H = float(self.height)
        if H <= 1e-12:
            return Rt
        t = float(np.clip(z / H, 0.0, 1.0))
        return (1.0 - t) * R0 + t * Rt


@dataclass(frozen=True)
class RidgeParams:
    """Parameters for external spiral/helical ridges."""
    n_ridges: int = 4
    pitch: float = 2.0                 # base pitch: z distance per one revolution
    pitch_top: Optional[float] = None  # if provided: linearly vary pitch from base->top
    ridge_height: float = 0.06         # base radial offset (purely visual)
    ridge_height_top: Optional[float] = None  # if provided: vary ridge_height base->top
    start_z_frac: float = 0.0          # ridges exist only on [start_z_frac*H, end_z_frac*H]
    end_z_frac: float = 1.0
    handedness: RidgeHandedness = "both"


@dataclass(frozen=True)
class SievePlateParams:
    """Parameters for the terminal sieve plate (cap)."""
    n_rings: int = 6
    n_spokes: int = 24
    z_at_top: Optional[float] = None   # defaults to lattice height
    dome_height: float = 0.0           # >0 makes the sieve plate convex (schematic)


@dataclass(frozen=True)
class AnchorBundleParams:
    """Parameters for a bundle of flexible anchoring spicules (root tuft)."""
    n_fibers: int = 80
    length: float = 2.5
    spread: float = 1.0
    waviness: float = 0.25
    seed: int = 0


@dataclass(frozen=True)
class SpiculeLamellaParams:
    """Concentric lamella model for a single spicule cross-section."""
    outer_radius: float = 1.0
    n_silica_layers: int = 18

    # In physical systems, organic interlayers can be nm-thick while silica layers are
    # ~0.1–2 µm thick. In this schematic model, thicknesses are in arbitrary units.
    organic_thickness: float = 0.01

    # graded silica lamellae (inner thick -> outer thin)
    silica_thickness_inner: float = 0.14
    silica_thickness_outer: float = 0.02

    axial_filament_radius: float = 0.05    # central proteinaceous filament/core

    thickness_profile: Literal["linear", "powerlaw"] = "linear"
    powerlaw_exp: float = 1.4              # used if profile="powerlaw"


@dataclass(frozen=True)
class CompositeBeamParams:
    """Bundle of spicules embedded in a silica matrix (cross-section)."""
    beam_radius: float = 1.0
    n_spicules: int = 20
    spicule_radius_mean: float = 0.12
    spicule_radius_std: float = 0.03
    packing_padding: float = 0.01
    seed: int = 0


@dataclass(frozen=True)
class CruciformSpiculeParams:
    """
    Non-planar cruciform (stauractine-like) spicule element (schematic).

    Many descriptions highlight:
    - long "vertical" rays vs shorter "horizontal" rays,
    - mild non-planarity (one ray tilted out of the primary plane).
    """
    ray_length: float = 1.0
    vertical_to_horizontal_ratio: float = 2.0
    tilt_degrees: float = 20.0
    n_points_per_ray: int = 30


@dataclass(frozen=True)
class NanoparticleRingParams:
    """Schematic silica nanoparticle rings in cross-section."""
    outer_radius: float = 1.0
    axial_filament_radius: float = 0.08
    n_rings: int = 10
    particles_per_ring: int = 40
    radial_jitter: float = 0.02
    angular_jitter: float = 0.04
    seed: int = 0

    # optional: scale marker size with radius (inner smaller -> outer larger)
    scale_marker_by_radius: bool = False
    marker_size_inner: float = 6.0
    marker_size_outer: float = 16.0

    # optional: illustrate "subparticles" around each particle (conceptual)
    subparticles_per_particle: int = 0
    subparticle_cloud_radius: float = 0.01


@dataclass(frozen=True)
class OpticalSpiculeParams:
    """
    Radial refractive-index profile of a basalia spicule (schematic, in microns).

    Typical qualitative structure:
    - proteinaceous axial filament (sub-micron)
    - high-index core region (~1–2 µm diameter)
    - lower-index central cylinder (~15–25 µm diameter)
    - outer shell with weak radial increase + oscillations (lamellae/organic bands)

    These parameters are *approximate* and intended for interactive exploration.
    """
    outer_radius_um: float = 50.0
    axial_filament_radius_um: float = 0.25
    core_radius_um: float = 0.75
    central_cylinder_radius_um: float = 10.0

    n_core_min: float = 1.45
    n_core_max: float = 1.48
    n_central_cylinder: float = 1.425
    n_shell_inner: float = 1.433
    n_shell_outer: float = 1.438

    shell_osc_amp: float = 0.0008
    shell_layer_spacing_um: float = 0.9

    def n_core(self) -> float:
        return 0.5 * (self.n_core_min + self.n_core_max)


@dataclass(frozen=True)
class BasaliaSpiculeParams:
    """Schematic basalia spicule with a smooth and a barbed region."""
    length: float = 3.0
    radius: float = 0.05

    # barbed region occupies [0, barbed_fraction*length] from the base
    barbed_fraction: float = 0.35
    n_barbs: int = 80
    barb_length: float = 0.08

    # optional terminal "apical spinous process" (schematic)
    tip_spines: int = 6
    tip_spine_length: float = 0.15

    waviness: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class IndentationFractureParams:
    """
    Simple indentation-fracture scaling model.

    Many indentation-cracking relations scale as:
        Pc ∝ Kc^4 / (E^2 * H)
    where Pc is a characteristic cracking load, Kc fracture toughness,
    E modulus, H hardness. Here we provide a *normalized* estimate.
    """
    Kc_MPa_sqrtm: float = 0.5
    E_GPa: float = 35.0
    H_GPa: float = 5.0


# -----------------------------
# Geometry: Euplectella-like lattice
# -----------------------------

def cylindrical_lattice_nodes_edges(p: LatticeParams) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build a cylindrical square grid with optional diagonal bracing.
    Supports linear taper (radius->radius_top) and an optional second, angularly-offset lattice.

    Returns:
        nodes: (N,3) float array
        edges: list of (i,j) index pairs into nodes
    """
    n_theta = max(3, int(p.n_theta))
    n_z = max(2, int(p.n_z))
    H = float(p.height)

    thetas = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    zs = np.linspace(0.0, H, n_z)

    def build_one_lattice(theta_offset: float = 0.0, radial_offset: float = 0.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        nodes_list = []
        for z in zs:
            Rz = p.radius_at(z) + radial_offset
            for th in thetas + theta_offset:
                nodes_list.append([Rz * math.cos(th), Rz * math.sin(th), z])
        nodes_arr = np.asarray(nodes_list, dtype=float)

        def idx(i: int, j: int) -> int:
            return (j * n_theta) + (i % n_theta)

        edges: List[Tuple[int, int]] = []

        # Vertical struts
        for j in range(n_z - 1):
            for i in range(n_theta):
                edges.append((idx(i, j), idx(i, j + 1)))

        # Horizontal rings
        for j in range(n_z):
            for i in range(n_theta):
                if p.seam or i < n_theta - 1:
                    edges.append((idx(i, j), idx(i + 1, j)))

        # Diagonals
        if p.diagonal_mode != "none":
            for j in range(n_z - 1):
                for i in range(n_theta):
                    cell_is_active = True
                    if p.diagonal_mode == "checkerboard":
                        cell_is_active = ((i + j) % 2 == 0)
                    if cell_is_active:
                        edges.append((idx(i, j), idx(i + 1, j + 1)))
                        edges.append((idx(i + 1, j), idx(i, j + 1)))

        return nodes_arr, edges

    # Primary lattice
    nodes1, edges1 = build_one_lattice(theta_offset=0.0, radial_offset=0.0)

    if not p.interpenetrating:
        return nodes1, edges1

    # Secondary lattice (angularly offset by fraction of theta step)
    dtheta = 2 * math.pi / n_theta
    theta_offset = float(p.secondary_theta_offset) * dtheta
    nodes2, edges2_local = build_one_lattice(theta_offset=theta_offset, radial_offset=float(p.secondary_radial_offset))

    # Combine
    nodes = np.vstack([nodes1, nodes2])

    offset = nodes1.shape[0]
    edges2 = [(a + offset, b + offset) for (a, b) in edges2_local]
    edges = edges1 + edges2
    return nodes, edges


def paired_diagonal_edges(nodes: np.ndarray, edges: List[Tuple[int, int]], offset: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Duplicate diagonal edges with a tiny radial offset to mimic "paired" diagonals.
    The offset is purely visual; it creates new nodes along the diagonal endpoints.

    If offset <= 0: returns input unchanged.
    """
    if offset <= 0:
        return nodes, edges

    new_nodes = nodes.tolist()
    new_edges: List[Tuple[int, int]] = []

    for (a, b) in edges:
        da = nodes[b] - nodes[a]
        # heuristic: diagonal if both z and x/y change
        is_diag = (abs(da[2]) > 1e-9) and (math.hypot(da[0], da[1]) > 1e-9)
        new_edges.append((a, b))
        if not is_diag:
            continue

        # create offset copies of the two endpoints
        for k in [a, b]:
            x, y, z = nodes[k]
            r = math.hypot(x, y)
            if r < 1e-12:
                dx, dy = 0.0, 0.0
            else:
                dx, dy = (x / r) * offset, (y / r) * offset
            new_nodes.append([x + dx, y + dy, z])
        a2 = len(new_nodes) - 2
        b2 = len(new_nodes) - 1
        new_edges.append((a2, b2))

    return np.asarray(new_nodes, dtype=float), new_edges


# -----------------------------
# Geometry: external spiral ridges (helices)
# -----------------------------

def helical_ridges(
    radius: float,
    height: float,
    p: RidgeParams,
    n_points: int = 400,
    radius_top: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Returns a list of polylines (each polyline is (n_points,3) array)
    representing helical ridges.

    Supports:
    - ridges only over a z-window (start_z_frac .. end_z_frac)
    - linear pitch gradient (pitch -> pitch_top)
    - linear ridge-height gradient (ridge_height -> ridge_height_top)
    - tapered radius (radius -> radius_top)
    """
    R0 = float(radius)
    Rt = float(radius_top) if radius_top is not None else R0
    H = float(height)

    # clamp z window
    z0 = float(np.clip(p.start_z_frac, 0.0, 1.0)) * H
    z1 = float(np.clip(p.end_z_frac, 0.0, 1.0)) * H
    if z1 <= z0 + 1e-12:
        return []

    n_ridges = max(1, int(p.n_ridges))
    base_phases = np.linspace(0, 2 * math.pi, n_ridges, endpoint=False)

    pitch0 = float(max(1e-6, p.pitch))
    pitch1 = float(max(1e-6, p.pitch_top)) if p.pitch_top is not None else pitch0

    h0 = float(max(0.0, p.ridge_height))
    h1 = float(max(0.0, p.ridge_height_top)) if p.ridge_height_top is not None else h0

    zs = np.linspace(z0, z1, int(max(10, n_points)))

    # radius + ridge-height profiles along z
    t = (zs - z0) / (z1 - z0)
    Rz = (1.0 - t) * (R0 + (Rt - R0) * (z0 / H if H > 1e-12 else 0.0)) + t * (R0 + (Rt - R0) * (z1 / H if H > 1e-12 else 1.0))
    # More simply: compute radius via linear taper:
    Rz = (1.0 - (zs / H if H > 1e-12 else 0.0)) * R0 + (zs / H if H > 1e-12 else 1.0) * Rt
    Hz = (1.0 - t) * h0 + t * h1

    # pitch profile along z
    pitch_z = (1.0 - t) * pitch0 + t * pitch1
    # dtheta/dz = 2π / pitch(z)
    dtheta_dz = (2 * math.pi) / np.clip(pitch_z, 1e-6, None)

    # integrate theta(z) using cumulative sum
    dz = np.gradient(zs)
    theta_incr = dtheta_dz * dz
    theta_cum = np.cumsum(theta_incr)
    theta_cum -= theta_cum[0]

    ridges: List[np.ndarray] = []

    def make(sign: float) -> None:
        for phase in base_phases:
            theta = phase + sign * theta_cum
            rr = Rz + Hz
            x = rr * np.cos(theta)
            y = rr * np.sin(theta)
            z = zs
            ridges.append(np.stack([x, y, z], axis=1))

    if p.handedness in ("right", "both"):
        make(+1.0)
    if p.handedness in ("left", "both"):
        make(-1.0)

    return ridges


# -----------------------------
# Geometry: terminal sieve plate (cap)
# -----------------------------

def sieve_plate_polylines(radius: float, p: SievePlateParams) -> List[np.ndarray]:
    """
    Returns polylines for concentric rings + spokes at z=z_top.
    If p.dome_height > 0, rings/spokes are lifted to form a convex dome.
    """
    R = float(radius)
    n_rings = max(1, int(p.n_rings))
    n_spokes = max(3, int(p.n_spokes))
    z_top = float(p.z_at_top) if p.z_at_top is not None else 0.0
    dome = float(p.dome_height)

    rings: List[np.ndarray] = []
    thetas = np.linspace(0, 2 * math.pi, 241)

    def z_dome(r: float) -> float:
        if dome <= 0:
            return z_top
        # simple paraboloid dome: highest at center
        return z_top + dome * (1.0 - (r / max(1e-9, R)) ** 2)

    # Rings
    for k in range(1, n_rings + 1):
        r = (k / (n_rings + 1)) * R
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        z = np.full_like(x, z_dome(r))
        rings.append(np.stack([x, y, z], axis=1))

    # Spokes
    spokes: List[np.ndarray] = []
    angles = np.linspace(0, 2 * math.pi, n_spokes, endpoint=False)
    for a in angles:
        rs = np.linspace(0, R, 80)
        x = rs * np.cos(a)
        y = rs * np.sin(a)
        z = np.array([z_dome(r) for r in rs], dtype=float)
        spokes.append(np.stack([x, y, z], axis=1))

    return rings + spokes


# -----------------------------
# Micro: laminated spicule radii
# -----------------------------

def spicule_layer_boundaries(p: SpiculeLamellaParams) -> Dict[str, Any]:
    """
    Compute inner/outer radii of successive silica lamellae separated by organic interlayers.

    Returns a dict with:
      - layers: list of (r_inner, r_outer, "silica"/"organic") from core outward
      - axial_filament_radius
      - outer_radius
    """
    R = float(p.outer_radius)
    n = max(1, int(p.n_silica_layers))
    t_org = float(max(0.0, p.organic_thickness))

    # Build silica thickness profile (inner thick -> outer thin)
    t_in = float(max(1e-6, p.silica_thickness_inner))
    t_out = float(max(1e-6, p.silica_thickness_outer))

    if p.thickness_profile == "linear":
        silica_ts = np.linspace(t_in, t_out, n)
    else:
        exp = float(max(0.1, p.powerlaw_exp))
        # map k in [0..n-1] to u in [0..1], then thickness = t_out + (t_in-t_out)*(1-u)^exp
        u = np.linspace(0.0, 1.0, n)
        silica_ts = t_out + (t_in - t_out) * (1.0 - u) ** exp

    r0 = float(max(0.0, p.axial_filament_radius))
    layers: List[Tuple[float, float, str]] = []
    r = r0
    for k in range(n):
        # silica
        a = r
        b = r + float(silica_ts[k])
        layers.append((a, b, "silica"))
        r = b
        # organic, except after last layer
        if k != n - 1 and t_org > 0:
            a = r
            b = r + t_org
            layers.append((a, b, "organic"))
            r = b

    # clip to outer radius
    clipped = []
    for (a, b, kind) in layers:
        if a >= R:
            break
        clipped.append((a, min(b, R), kind))

    return {"layers": clipped, "axial_filament_radius": r0, "outer_radius": R}


# -----------------------------
# Geometry: composite strut (bundled spicules) cross-section
# -----------------------------

def pack_circles_in_circle(
    R: float,
    radii: np.ndarray,
    padding: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 20000,
) -> np.ndarray:
    """
    Randomly pack circles (with given radii) inside a larger circle radius R.
    Returns centers (N,2). Very simple rejection sampling; enough for schematic.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    centers = []
    for rr in radii:
        placed = False
        for _ in range(max_tries):
            # sample a point inside the big circle, leaving margin rr
            rmax = max(1e-9, R - rr - padding)
            u = rng.random()
            v = rng.random()
            rad = math.sqrt(u) * rmax
            ang = 2 * math.pi * v
            x = rad * math.cos(ang)
            y = rad * math.sin(ang)
            ok = True
            for (cx, cy), rprev in centers:
                if math.hypot(x - cx, y - cy) < (rr + rprev + padding):
                    ok = False
                    break
            if ok:
                centers.append(((x, y), rr))
                placed = True
                break
        if not placed:
            # give up: still return what we have + an approximate location
            centers.append(((0.0, 0.0), rr))
    return np.asarray([c for (c, _) in centers], dtype=float)


# -----------------------------
# Macro: anchor bundle curves
# -----------------------------

def anchor_bundle_curves(p: AnchorBundleParams) -> List[np.ndarray]:
    """
    Generate a set of 3D wavy curves anchored near (0,0,0) and extending downward.
    """
    rng = np.random.default_rng(int(p.seed))
    curves: List[np.ndarray] = []
    n = max(1, int(p.n_fibers))
    L = float(p.length)
    spread = float(p.spread)
    wav = float(p.waviness)

    for _ in range(n):
        # random direction in x-y for base offset
        ang = 2 * math.pi * rng.random()
        rad = spread * math.sqrt(rng.random())
        x0 = rad * math.cos(ang)
        y0 = rad * math.sin(ang)
        # create a curve
        t = np.linspace(0, 1, 60)
        z = -L * t
        # gentle waviness
        ph1 = 2 * math.pi * rng.random()
        ph2 = 2 * math.pi * rng.random()
        x = x0 + wav * 0.3 * np.sin(2 * math.pi * t + ph1) + wav * 0.15 * np.sin(4 * math.pi * t + ph2)
        y = y0 + wav * 0.3 * np.cos(2 * math.pi * t + ph2) + wav * 0.15 * np.cos(4 * math.pi * t + ph1)
        curves.append(np.stack([x, y, z], axis=1))
    return curves


# -----------------------------
# Micro: cruciform spicule element
# -----------------------------

def cruciform_spicule(p: CruciformSpiculeParams) -> np.ndarray:
    """
    Return a (N,3) polyline list concatenated with NaNs separating rays.
    """
    Lh = float(p.ray_length)
    Lv = float(p.ray_length) * float(max(0.1, p.vertical_to_horizontal_ratio))
    tilt = math.radians(float(p.tilt_degrees))
    n = max(2, int(p.n_points_per_ray))

    # Rays: two vertical (±z), two horizontal (±x) but one horizontal is tilted out of the x-z plane.
    rays = []

    # vertical up/down
    z_up = np.linspace(0, Lv, n)
    rays.append(np.stack([np.zeros_like(z_up), np.zeros_like(z_up), z_up], axis=1))
    z_dn = np.linspace(0, -Lv, n)
    rays.append(np.stack([np.zeros_like(z_dn), np.zeros_like(z_dn), z_dn], axis=1))

    # horizontal +x (in plane y=0)
    x1 = np.linspace(0, Lh, n)
    rays.append(np.stack([x1, np.zeros_like(x1), np.zeros_like(x1)], axis=1))

    # "non-planar" horizontal ray: mostly -x, but with +y component to tilt out of plane
    s = np.linspace(0, 1, n)
    x2 = -Lh * np.cos(tilt) * s
    y2 = +Lh * np.sin(tilt) * s
    z2 = np.zeros_like(s)
    rays.append(np.stack([x2, y2, z2], axis=1))

    # concatenate with NaN separators for Plotly
    sep = np.full((1, 3), np.nan)
    poly = []
    for r in rays:
        poly.append(r)
        poly.append(sep)
    return np.vstack(poly)


# -----------------------------
# Nano: nanoparticle rings
# -----------------------------

def nanoparticle_ring_points(p: NanoparticleRingParams) -> np.ndarray:
    rng = np.random.default_rng(int(p.seed))
    R = float(p.outer_radius)
    r0 = float(max(0.0, p.axial_filament_radius))
    n_rings = max(1, int(p.n_rings))
    k = max(3, int(p.particles_per_ring))

    rs = np.linspace(r0 + (R - r0) / (n_rings + 1), R, n_rings)
    pts = []
    for rr in rs:
        thetas = np.linspace(0, 2 * math.pi, k, endpoint=False)
        for th in thetas:
            rj = rr + float(rng.normal(0.0, p.radial_jitter))
            tj = th + float(rng.normal(0.0, p.angular_jitter))
            pts.append([rj * math.cos(tj), rj * math.sin(tj)])
    return np.asarray(pts, dtype=float)


def _nanoparticle_subcloud(centers: np.ndarray, n_sub: int, cloud_r: float, seed: int) -> np.ndarray:
    if n_sub <= 0:
        return np.zeros((0, 2), dtype=float)
    rng = np.random.default_rng(seed)
    pts = []
    for (cx, cy) in centers:
        # scatter points in a small disk around center
        u = rng.random(n_sub)
        v = rng.random(n_sub)
        rr = np.sqrt(u) * cloud_r
        th = 2 * math.pi * v
        x = cx + rr * np.cos(th)
        y = cy + rr * np.sin(th)
        pts.append(np.stack([x, y], axis=1))
    return np.vstack(pts) if pts else np.zeros((0, 2), dtype=float)


# -----------------------------
# Optics: basalia spicule (index profile)
# -----------------------------

def optical_spicule_refractive_index(p: OpticalSpiculeParams, r_um: np.ndarray) -> np.ndarray:
    """
    Piecewise refractive index profile n(r) for a basalia spicule cross-section.
    """
    r = np.asarray(r_um, dtype=float)

    n = np.full_like(r, p.n_shell_outer)

    # regions
    r_af = float(p.axial_filament_radius_um)
    r_core = float(p.core_radius_um)
    r_cc = float(p.central_cylinder_radius_um)
    r_out = float(p.outer_radius_um)

    # outside outer radius: leave as outer
    # shell region: r_cc .. r_out, linear increase + oscillation
    shell_mask = (r >= r_cc) & (r <= r_out)
    if np.any(shell_mask):
        t = (r[shell_mask] - r_cc) / max(1e-9, (r_out - r_cc))
        n_shell = (1.0 - t) * float(p.n_shell_inner) + t * float(p.n_shell_outer)
        # weak oscillation (organic bands / lamellae)
        if p.shell_osc_amp != 0 and p.shell_layer_spacing_um > 1e-9:
            n_shell = n_shell + float(p.shell_osc_amp) * np.sin(2 * math.pi * (r[shell_mask] - r_cc) / float(p.shell_layer_spacing_um))
        n[shell_mask] = n_shell

    # central cylinder: r_core .. r_cc
    cc_mask = (r >= r_core) & (r < r_cc)
    n[cc_mask] = float(p.n_central_cylinder)

    # high-index core: r_af .. r_core (let it vary slightly from min->max across radius)
    core_mask = (r >= r_af) & (r < r_core)
    if np.any(core_mask):
        tt = (r[core_mask] - r_af) / max(1e-9, (r_core - r_af))
        n[core_mask] = (1.0 - tt) * float(p.n_core_min) + tt * float(p.n_core_max)

    # axial filament: assign a lower effective index (mostly organic); schematic
    af_mask = r < r_af
    n[af_mask] = float(p.n_central_cylinder)

    return n


def optical_spicule_metrics(p: OpticalSpiculeParams, wavelength_um: float = 0.633) -> Dict[str, float]:
    """
    Basic step-index-like derived quantities for intuition (approximate):
    - NA using n_core_max vs n_central_cylinder
    - acceptance half-angle in air
    - V-number using core radius (core_radius_um)
    """
    n1 = float(p.n_core_max)
    n2 = float(p.n_central_cylinder)
    NA = math.sqrt(max(0.0, n1 ** 2 - n2 ** 2))
    theta_air = math.degrees(math.asin(min(1.0, NA / 1.0)))  # n_air ~ 1
    a = float(p.core_radius_um)
    lam = float(max(1e-6, wavelength_um))
    V = (2 * math.pi * a / lam) * NA
    return {"NA": NA, "theta_air_deg": theta_air, "V": V}


# -----------------------------
# Holdfast: basalia spicule with barbs
# -----------------------------

def basalia_spicule_segments(p: BasaliaSpiculeParams) -> Dict[str, Any]:
    """
    Return a dict with:
      - 'axis': (N,3) polyline
      - 'barbs': list of (2,3) segments
      - 'tip_spines': list of (2,3) segments
    """
    rng = np.random.default_rng(int(p.seed))
    L = float(p.length)
    r = float(p.radius)
    wav = float(p.waviness)

    # main axis curve (mostly along -z)
    t = np.linspace(0, 1, 160)
    z = -L * t
    # optional waviness in x/y
    x = wav * 0.1 * np.sin(2 * math.pi * t + 2 * math.pi * rng.random())
    y = wav * 0.1 * np.cos(2 * math.pi * t + 2 * math.pi * rng.random())
    axis = np.stack([x, y, z], axis=1)

    # barbs on proximal region (near base: t in [0, barbed_fraction])
    bf = float(np.clip(p.barbed_fraction, 0.0, 1.0))
    n_barbs = max(0, int(p.n_barbs))
    barb_L = float(max(0.0, p.barb_length))
    barbs = []
    if n_barbs > 0 and barb_L > 0 and bf > 0:
        t_barb = rng.random(n_barbs) * bf
        for tb in t_barb:
            # interpolate center position along axis
            idx = int(tb * (len(axis) - 1))
            cx, cy, cz = axis[idx]
            ang = 2 * math.pi * rng.random()
            # barb starts on surface and points outward
            sx = cx + r * math.cos(ang)
            sy = cy + r * math.sin(ang)
            sz = cz
            ex = cx + (r + barb_L) * math.cos(ang)
            ey = cy + (r + barb_L) * math.sin(ang)
            ez = cz
            barbs.append(np.asarray([[sx, sy, sz], [ex, ey, ez]], dtype=float))

    # tip spines at the distal end (most negative z)
    tip_spines = []
    n_tip = max(0, int(p.tip_spines))
    tip_L = float(max(0.0, p.tip_spine_length))
    if n_tip > 0 and tip_L > 0:
        cx, cy, cz = axis[-1]
        for k in range(n_tip):
            ang = 2 * math.pi * (k / n_tip)
            sx = cx + r * math.cos(ang)
            sy = cy + r * math.sin(ang)
            sz = cz
            ex = cx + (r + tip_L) * math.cos(ang)
            ey = cy + (r + tip_L) * math.sin(ang)
            ez = cz
            tip_spines.append(np.asarray([[sx, sy, sz], [ex, ey, ez]], dtype=float))

    return {"axis": axis, "barbs": barbs, "tip_spines": tip_spines}


# -----------------------------
# Indentation fracture scaling
# -----------------------------

def normalized_cracking_load(p: IndentationFractureParams) -> float:
    """Return a normalized cracking load Pc ~ Kc^4 / (E^2 H). Units cancel for relative comparisons."""
    Kc = float(max(1e-9, p.Kc_MPa_sqrtm))
    E = float(max(1e-9, p.E_GPa))
    H = float(max(1e-9, p.H_GPa))
    return (Kc ** 4) / (E ** 2 * H)


# -----------------------------
# Plotly helpers
# -----------------------------

def _segments_to_scatter3d(segments: List[np.ndarray], name: str, mode: str = "lines") -> "go.Scatter3d":
    xs, ys, zs = [], [], []
    for seg in segments:
        xs.extend(seg[:, 0].tolist())
        ys.extend(seg[:, 1].tolist())
        zs.extend(seg[:, 2].tolist())
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode=mode, name=name)


def _edges_to_scatter3d(nodes: np.ndarray, edges: List[Tuple[int, int]], name: str = "edges") -> "go.Scatter3d":
    xs, ys, zs = [], [], []
    for (a, b) in edges:
        xa, ya, za = nodes[a]
        xb, yb, zb = nodes[b]
        xs.extend([xa, xb, None])
        ys.extend([ya, yb, None])
        zs.extend([za, zb, None])
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name)


# -----------------------------
# Figures
# -----------------------------

def lattice_figure(
    lattice: LatticeParams,
    ridges: Optional[RidgeParams] = None,
    sieve: Optional[SievePlateParams] = None,
    show_axes: bool = False,
) -> "go.Figure":
    """
    Plotly 3D visualization of a cylindrical lattice with optional ridges + sieve plate.
    """
    nodes, edges = cylindrical_lattice_nodes_edges(lattice)
    if lattice.diagonal_pair_offset > 0:
        nodes, edges = paired_diagonal_edges(nodes, edges, lattice.diagonal_pair_offset)

    traces = [_edges_to_scatter3d(nodes, edges, name="lattice")]

    # Ridges
    if ridges is not None:
        ridge_lines = helical_ridges(
            radius=lattice.radius,
            height=lattice.height,
            p=ridges,
            radius_top=lattice.radius_top,
        )
        if ridge_lines:
            traces.append(_segments_to_scatter3d(ridge_lines, name="ridges"))

    # Sieve plate (use top radius)
    if sieve is not None:
        z_top = lattice.height if sieve.z_at_top is None else sieve.z_at_top
        sieve2 = SievePlateParams(n_rings=sieve.n_rings, n_spokes=sieve.n_spokes, z_at_top=z_top, dome_height=sieve.dome_height)
        R_top = lattice.radius_at(float(z_top))
        polylines = sieve_plate_polylines(R_top, sieve2)
        traces.append(_segments_to_scatter3d(polylines, name="sieve plate"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=show_axes),
            yaxis=dict(visible=show_axes),
            zaxis=dict(visible=show_axes),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Euplectella-like cylindrical lattice (schematic)",
        showlegend=True,
    )
    return fig


def full_skeleton_figure(
    lattice: LatticeParams,
    ridges: Optional[RidgeParams],
    sieve: Optional[SievePlateParams],
    anchor: AnchorBundleParams,
    show_axes: bool = False,
) -> "go.Figure":
    """
    Combine lattice + optional ridges + optional sieve plate + anchor bundle into one schematic.
    """
    fig = lattice_figure(lattice=lattice, ridges=ridges, sieve=sieve, show_axes=show_axes)

    # Anchor bundle: place at base center and extend downward
    curves = anchor_bundle_curves(anchor)
    anchor_trace = _segments_to_scatter3d(curves, name="anchor bundle")
    fig.add_trace(anchor_trace)

    fig.update_layout(title="Glass sponge skeletal motifs (macro schematic)")
    return fig


def spicule_cross_section_figure(p: SpiculeLamellaParams, show_legend: bool = True) -> "go.Figure":
    """
    2D cross-section of a laminated spicule as concentric circles.
    """
    info = spicule_layer_boundaries(p)
    layers = info["layers"]
    R = info["outer_radius"]

    fig = go.Figure()

    # Draw boundaries as circles; organic layers dashed
    thetas = np.linspace(0, 2 * math.pi, 361)

    def circle(r: float):
        return r * np.cos(thetas), r * np.sin(thetas)

    # outer boundary
    x, y = circle(R)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="outer boundary"))

    for (a, b, kind) in layers:
        x, y = circle(b)
        dash = "dash" if kind == "organic" else "solid"
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=kind, line=dict(dash=dash)))

    fig.update_layout(
        title="Laminated spicule cross-section (schematic)",
        xaxis=dict(scaleanchor="y", zeroline=False, visible=False),
        yaxis=dict(zeroline=False, visible=False),
        showlegend=show_legend,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def composite_beam_cross_section_figure(p: CompositeBeamParams, show_legend: bool = True) -> "go.Figure":
    """
    2D cross-section of a composite strut (bundle of spicules in matrix).
    """
    rng = np.random.default_rng(int(p.seed))
    R = float(p.beam_radius)
    n = max(1, int(p.n_spicules))
    radii = rng.normal(float(p.spicule_radius_mean), float(p.spicule_radius_std), size=n)
    radii = np.clip(radii, 0.01 * R, 0.35 * R)

    centers = pack_circles_in_circle(R, radii, padding=float(p.packing_padding), rng=rng)

    fig = go.Figure()
    thetas = np.linspace(0, 2 * math.pi, 241)

    # outer boundary
    fig.add_trace(go.Scatter(x=R * np.cos(thetas), y=R * np.sin(thetas), mode="lines", name="beam boundary"))

    # spicule circles
    for (cx, cy), rr in zip(centers, radii):
        x = cx + rr * np.cos(thetas)
        y = cy + rr * np.sin(thetas)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="spicule"))

    fig.update_layout(
        title="Composite strut cross-section (schematic)",
        xaxis=dict(scaleanchor="y", zeroline=False, visible=False),
        yaxis=dict(zeroline=False, visible=False),
        showlegend=show_legend,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def anchor_bundle_figure(p: AnchorBundleParams, show_axes: bool = False) -> "go.Figure":
    curves = anchor_bundle_curves(p)
    fig = go.Figure(data=[_segments_to_scatter3d(curves, name="anchor bundle")])
    fig.update_layout(
        title="Anchor spicule bundle (schematic)",
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=show_axes),
            yaxis=dict(visible=show_axes),
            zaxis=dict(visible=show_axes),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
    )
    return fig


def cruciform_spicule_figure(p: CruciformSpiculeParams, show_axes: bool = True) -> "go.Figure":
    poly = cruciform_spicule(p)
    fig = go.Figure(data=[go.Scatter3d(x=poly[:, 0], y=poly[:, 1], z=poly[:, 2], mode="lines", name="cruciform")])
    fig.update_layout(
        title="Cruciform spicule element (schematic)",
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=show_axes),
            yaxis=dict(visible=show_axes),
            zaxis=dict(visible=show_axes),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
    )
    return fig


def nanoparticle_rings_figure(p: NanoparticleRingParams, show_legend: bool = True) -> "go.Figure":
    pts = nanoparticle_ring_points(p)
    x = pts[:, 0]
    y = pts[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)

    if p.scale_marker_by_radius and len(r) > 0:
        rmin = float(np.min(r))
        rmax = float(np.max(r))
        if rmax <= rmin + 1e-12:
            sizes = np.full_like(r, float(p.marker_size_inner))
        else:
            t = (r - rmin) / (rmax - rmin)
            sizes = (1.0 - t) * float(p.marker_size_inner) + t * float(p.marker_size_outer)
    else:
        sizes = float(p.marker_size_inner)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="nanoparticles",
        marker=dict(size=sizes, opacity=0.8)
    ))

    # Optional subparticle clouds
    n_sub = int(p.subparticles_per_particle)
    if n_sub > 0 and float(p.subparticle_cloud_radius) > 0:
        sub = _nanoparticle_subcloud(pts, n_sub=n_sub, cloud_r=float(p.subparticle_cloud_radius), seed=int(p.seed) + 123)
        fig.add_trace(go.Scatter(
            x=sub[:, 0], y=sub[:, 1], mode="markers", name="subparticles",
            marker=dict(size=max(1.0, float(p.marker_size_inner) * 0.35), opacity=0.5)
        ))

    fig.update_layout(
        title="Silica nanoparticle rings (schematic)",
        xaxis=dict(scaleanchor="y", zeroline=False, visible=False),
        yaxis=dict(zeroline=False, visible=False),
        showlegend=show_legend,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def optical_spicule_index_profile_figure(p: OpticalSpiculeParams) -> "go.Figure":
    r = np.linspace(0, float(p.outer_radius_um), 600)
    n = optical_spicule_refractive_index(p, r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=n, mode="lines", name="n(r)"))
    fig.update_layout(
        title="Basalia spicule refractive-index profile (schematic)",
        xaxis_title="Radius r (µm)",
        yaxis_title="Refractive index n",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def optical_spicule_index_map_figure(p: OpticalSpiculeParams, n_grid: int = 220) -> "go.Figure":
    R = float(p.outer_radius_um)
    xs = np.linspace(-R, R, int(max(50, n_grid)))
    ys = np.linspace(-R, R, int(max(50, n_grid)))
    X, Y = np.meshgrid(xs, ys)
    rr = np.sqrt(X ** 2 + Y ** 2)
    n = optical_spicule_refractive_index(p, rr)
    n = np.where(rr <= R, n, np.nan)

    fig = go.Figure(data=go.Heatmap(x=xs, y=ys, z=n, colorscale="Viridis", colorbar=dict(title="n")))
    fig.update_layout(
        title="Basalia spicule index map (cross-section, schematic)",
        xaxis=dict(scaleanchor="y", visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def basalia_spicule_figure(p: BasaliaSpiculeParams, show_axes: bool = False) -> "go.Figure":
    info = basalia_spicule_segments(p)
    axis = info["axis"]
    barbs = info["barbs"]
    tip_spines = info["tip_spines"]

    traces = [go.Scatter3d(x=axis[:, 0], y=axis[:, 1], z=axis[:, 2], mode="lines", name="axis")]
    if barbs:
        traces.append(_segments_to_scatter3d(barbs, name="barbs"))
    if tip_spines:
        traces.append(_segments_to_scatter3d(tip_spines, name="tip spines"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Basalia spicule (smooth + barbed region, schematic)",
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=show_axes),
            yaxis=dict(visible=show_axes),
            zaxis=dict(visible=show_axes),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
    )
    return fig

