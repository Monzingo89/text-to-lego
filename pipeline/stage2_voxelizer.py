"""
stage2_voxelizer.py
===================
Stage 2: Structured Geometry JSON -> 3D NumPy Voxel Grid

Converts the geometric primitive spec from Stage 1 into a 3D NumPy array where:
  - Each cell is 0 (empty) or a positive integer (color index)
  - X axis = left-to-right (studs)
  - Y axis = front-to-back (studs)
  - Z axis = bottom-to-top (plates, 1 plate = 1/3 stud height)

Supported primitive shapes:
  box      - solid rectangular fill
  cylinder - circular cross-section extruded vertically
  cone     - cylinder that tapers from base_radius to 0
  arch     - box with a cylindrical void cut from the bottom
  strut    - Bresenham 3D line segment with configurable thickness
  lattice  - sparse strut network (diagonal cross-bracing pattern)
"""

import numpy as np
from typing import Any


# ─── Color management ─────────────────────────────────────────────────────────

def build_color_map(color_palette: dict) -> dict[str, int]:
    """Convert color palette dict to name->index mapping (1-based, 0=empty)."""
    color_map = {}
    for idx, name in enumerate(color_palette.keys(), start=1):
        color_map[name] = idx
    return color_map


def color_index(color_name: str, color_map: dict[str, int]) -> int:
    """Get color index, defaulting to 1 if not found."""
    return color_map.get(color_name, 1)


# ─── Primitive voxelizers ─────────────────────────────────────────────────────

def voxelize_box(grid: np.ndarray, origin: list, size: list, c: int) -> None:
    """Fill a rectangular box with color c."""
    x0, y0, z0 = int(origin[0]), int(origin[1]), int(origin[2])
    w, d, h = max(1, int(size[0])), max(1, int(size[1])), max(1, int(size[2]))
    x1 = min(x0 + w, grid.shape[0])
    y1 = min(y0 + d, grid.shape[1])
    z1 = min(z0 + h, grid.shape[2])
    if x0 < x1 and y0 < y1 and z0 < z1:
        grid[x0:x1, y0:y1, z0:z1] = c


def voxelize_cylinder(grid: np.ndarray, origin: list, size: list, c: int) -> None:
    """Fill a cylinder with circular XY cross-section."""
    x0, y0, z0 = int(origin[0]), int(origin[1]), int(origin[2])
    w, d, h = max(1, int(size[0])), max(1, int(size[1])), max(1, int(size[2]))
    cx = x0 + w / 2.0
    cy = y0 + d / 2.0
    rx = w / 2.0
    ry = d / 2.0
    z1 = min(z0 + h, grid.shape[2])
    for z in range(z0, z1):
        for x in range(max(0, x0), min(x0 + w, grid.shape[0])):
            for y in range(max(0, y0), min(y0 + d, grid.shape[1])):
                if rx > 0 and ry > 0:
                    if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                        grid[x, y, z] = c


def voxelize_cone(grid: np.ndarray, origin: list, size: list, c: int) -> None:
    """Fill a cone tapering from full base radius at z0 to point at z0+h."""
    x0, y0, z0 = int(origin[0]), int(origin[1]), int(origin[2])
    w, d, h = max(1, int(size[0])), max(1, int(size[1])), max(1, int(size[2]))
    cx = x0 + w / 2.0
    cy = y0 + d / 2.0
    z1 = min(z0 + h, grid.shape[2])
    for z in range(z0, z1):
        progress = (z - z0) / h  # 0 at base, 1 at tip
        scale = 1.0 - progress
        rx = (w / 2.0) * scale
        ry = (d / 2.0) * scale
        for x in range(max(0, x0), min(x0 + w, grid.shape[0])):
            for y in range(max(0, y0), min(y0 + d, grid.shape[1])):
                if rx > 0 and ry > 0:
                    if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                        grid[x, y, z] = c


def voxelize_arch(grid: np.ndarray, origin: list, size: list, c: int) -> None:
    """Fill an arch: box with a semi-cylindrical void cut from the bottom-center."""
    voxelize_box(grid, origin, size, c)
    x0, y0, z0 = int(origin[0]), int(origin[1]), int(origin[2])
    w, d, h = max(1, int(size[0])), max(1, int(size[1])), max(1, int(size[2]))
    arch_h = int(h * 0.6)
    arch_w = int(w * 0.5)
    arch_x0 = x0 + (w - arch_w) // 2
    cx = arch_x0 + arch_w / 2.0
    cy = y0 + d / 2.0
    rx = arch_w / 2.0
    ry = d / 2.0
    for z in range(z0, min(z0 + arch_h, grid.shape[2])):
        for x in range(max(0, arch_x0), min(arch_x0 + arch_w, grid.shape[0])):
            for y in range(max(0, y0), min(y0 + d, grid.shape[1])):
                if rx > 0 and ry > 0:
                    if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                        grid[x, y, z] = 0  # carve out


def _bresenham3d(x0, y0, z0, x1, y1, z1):
    """Yield voxel coordinates along a 3D line (Bresenham algorithm)."""
    points = []
    dx, dy, dz = abs(x1-x0), abs(y1-y0), abs(z1-z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1
    if dx >= dy and dx >= dz:
        p1, p2 = 2*dy - dx, 2*dz - dx
        while x0 != x1:
            points.append((x0, y0, z0))
            if p1 >= 0: y0 += sy; p1 -= 2*dx
            if p2 >= 0: z0 += sz; p2 -= 2*dx
            p1 += 2*dy; p2 += 2*dz; x0 += sx
    elif dy >= dx and dy >= dz:
        p1, p2 = 2*dx - dy, 2*dz - dy
        while y0 != y1:
            points.append((x0, y0, z0))
            if p1 >= 0: x0 += sx; p1 -= 2*dy
            if p2 >= 0: z0 += sz; p2 -= 2*dy
            p1 += 2*dx; p2 += 2*dz; y0 += sy
    else:
        p1, p2 = 2*dy - dz, 2*dx - dz
        while z0 != z1:
            points.append((x0, y0, z0))
            if p1 >= 0: y0 += sy; p1 -= 2*dz
            if p2 >= 0: x0 += sx; p2 -= 2*dz
            p1 += 2*dy; p2 += 2*dx; z0 += sz
    points.append((x1, y1, z1))
    return points


def voxelize_strut(grid: np.ndarray, origin: list, size: list, c: int,
                   angle_deg: float = 0.0, thickness: int = 1) -> None:
    """
    Draw a diagonal strut as a thick 3D line.
    angle_deg: degrees from vertical (0=straight up, 45=diagonal).
    The strut runs from origin to origin+size, angled.
    """
    x0, y0, z0 = int(origin[0]), int(origin[1]), int(origin[2])
    h = max(1, int(size[2]))
    import math
    angle_rad = math.radians(angle_deg)
    x_offset = int(h * math.sin(angle_rad) * (size[0] / max(size[2], 1)))
    x1 = x0 + x_offset
    y1 = y0
    z1 = z0 + h
    
    line_pts = _bresenham3d(x0, y0, z0, x1, y1, z1)
    half_t = thickness // 2
    for (lx, ly, lz) in line_pts:
        for tx in range(-half_t, half_t + 1):
            for ty in range(-half_t, half_t + 1):
                nx, ny = lx + tx, ly + ty
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= lz < grid.shape[2]:
                    grid[nx, ny, lz] = c


def voxelize_lattice(grid: np.ndarray, origin: list, size: list, c: int,
                     fill_density: float = 0.3) -> None:
    """
    Draw a lattice/truss pattern: diagonal cross-bracing within the bounding box.
    fill_density controls how many struts to draw (0.1=sparse, 1.0=nearly solid).
    """
    x0, y0, z0 = int(origin[0]), int(origin[1]), int(origin[2])
    w, d, h = max(1, int(size[0])), max(1, int(size[1])), max(1, int(size[2]))
    
    spacing = max(1, int(1.0 / fill_density))
    
    for z in range(z0, min(z0 + h, grid.shape[2]), spacing):
        for x in range(x0, min(x0 + w, grid.shape[0]), 2):
            for y in range(y0, min(y0 + d, grid.shape[1])):
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    grid[x, y, z] = c
        z_end = min(z + spacing, z0 + h, grid.shape[2])
        for x in range(x0, min(x0 + w - 1, grid.shape[0])):
            pts = _bresenham3d(x, y0, z, x + 1, y0, z_end - 1)
            for (px, py, pz) in pts:
                if 0 <= px < grid.shape[0] and 0 <= py < grid.shape[1] and 0 <= pz < grid.shape[2]:
                    grid[px, py, pz] = c


# ─── Dispatcher ───────────────────────────────────────────────────────────────

_SHAPE_HANDLERS = {
    "box": voxelize_box,
    "cylinder": voxelize_cylinder,
    "cone": voxelize_cone,
    "arch": voxelize_arch,
    "strut": voxelize_strut,
    "lattice": voxelize_lattice,
}


# ─── Main voxelize function ───────────────────────────────────────────────────

def voxelize(spec: dict[str, Any], verbose: bool = False) -> tuple[np.ndarray, dict]:
    """
    Convert a Stage 1 geometry spec to a 3D NumPy voxel grid.
    
    Args:
        spec:    The structured geometry dict from decompose_text()
        verbose: Print progress info
        
    Returns:
        Tuple of (grid, color_map) where:
          grid:      np.ndarray shape (X, Y, Z) uint8, 0=empty, 1+=color
          color_map: dict mapping color name -> integer index
          
    Example:
        >>> spec = decompose_text("cube")
        >>> grid, colors = voxelize(spec)
        >>> grid.shape
        (10, 10, 30)
        >>> np.sum(grid > 0)  # filled voxels
        3000
    """
    target = spec.get("target_studs", [32, 32, 96])
    X, Y, Z = int(target[0]), int(target[1]), int(target[2])
    
    if verbose:
        print(f"[Stage 2] Creating voxel grid: {X}x{Y}x{Z} (X x Y x Z plates)")
    
    grid = np.zeros((X, Y, Z), dtype=np.uint8)
    
    color_palette = spec.get("color_palette", {"primary": [128, 128, 128]})
    color_map = build_color_map(color_palette)
    
    primitives = spec.get("primitives", [])
    
    for i, prim in enumerate(primitives):
        name = prim.get("name", f"primitive_{i}")
        shape = prim.get("shape", "box").lower()
        origin = prim.get("origin", [0, 0, 0])
        size = prim.get("size", [4, 4, 4])
        color_name = prim.get("color", "primary")
        c = color_index(color_name, color_map)
        fill_density = float(prim.get("fill_density", 1.0))
        angle_deg = float(prim.get("angle_deg") or 0.0)
        
        if verbose:
            print(f"  [{i+1}/{len(primitives)}] {name}: {shape} @ {origin} size {size} color {color_name}({c})")
        
        handler = _SHAPE_HANDLERS.get(shape, voxelize_box)
        
        try:
            if shape == "strut":
                handler(grid, origin, size, c, angle_deg=angle_deg)
            elif shape == "lattice":
                handler(grid, origin, size, c, fill_density=fill_density)
            else:
                handler(grid, origin, size, c)
        except Exception as e:
            if verbose:
                print(f"    WARNING: Failed to voxelize {name}: {e}")
    
    filled = int(np.sum(grid > 0))
    total = X * Y * Z
    if verbose:
        print(f"[Stage 2] Done. Filled: {filled}/{total} voxels ({100*filled/total:.1f}%)")
    
    return grid, color_map


def save_grid(grid: np.ndarray, path: str) -> None:
    """Save voxel grid to .npy file."""
    np.save(path, grid)
    print(f"Saved grid {grid.shape} to {path}")


def load_grid(path: str) -> np.ndarray:
    """Load voxel grid from .npy file."""
    return np.load(path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Stage 2: Geometry spec -> voxel grid")
    parser.add_argument("spec_file", help="Path to Stage 1 JSON output")
    parser.add_argument("--output", default="grid.npy")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    with open(args.spec_file) as f:
        spec = json.load(f)
    
    grid, color_map = voxelize(spec, verbose=args.verbose)
    save_grid(grid, args.output)
    print(f"Color map: {color_map}")
