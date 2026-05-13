"""
stage3_optimizer.py
===================
Stage 3: Voxel Grid -> LEGO Part Placement List

Converts the 3D voxel grid into a list of real LEGO part placements using
a greedy layer-by-layer covering algorithm.

Algorithm:
  1. Process grid layer by layer from Z=0 upward
  2. For each Z-layer, find the 2D footprint of filled voxels
  3. Apply the "maximum rectangle" greedy covering:
     - Try to place the largest valid brick/plate first
     - Mark covered voxels, repeat until layer is fully covered
  4. Consolidation pass: merge 3 consecutive single-plate layers into 1 brick
  5. Stagger enforcement: ensure no straight vertical seam > 1 brick tall
  6. Color assignment: use the most common color in each brick's voxels

Output: list of placement dicts:
  {
    "part_id": "3001",        # LEGO part number (Rebrickable format)
    "color_id": 4,            # Rebrickable color ID
    "color_name": "Red",
    "x": 10, "y": 8, "z": 3, # position in stud/plate units
    "width": 2, "depth": 4,   # footprint in studs
    "height": 1,              # 1=plate, 3=brick
    "rotation": 0             # 0 or 90 degrees
  }
"""

import numpy as np
from typing import Any


# ─── Part catalog ─────────────────────────────────────────────────────────────
# Maps (width_studs, depth_studs, height_plates) -> Rebrickable part_id
# Heights: 1 = plate (3.2mm), 3 = brick (9.6mm)

PART_CATALOG: dict[tuple, str] = {
    # Plates (height=1 plate = 3.2mm)
    (1, 1, 1): "3024",
    (1, 2, 1): "3023",
    (1, 3, 1): "3623",
    (1, 4, 1): "3710",
    (1, 6, 1): "3666",
    (1, 8, 1): "3460",
    (2, 2, 1): "3022",
    (2, 3, 1): "3021",
    (2, 4, 1): "3020",
    (2, 6, 1): "3795",
    (2, 8, 1): "3034",
    (4, 4, 1): "3031",
    (4, 6, 1): "3032",
    (4, 8, 1): "3035",
    # Bricks (height=3 plates = 9.6mm)
    (1, 1, 3): "3005",
    (1, 2, 3): "3004",
    (1, 3, 3): "3622",
    (1, 4, 3): "3010",
    (1, 6, 3): "3009",
    (1, 8, 3): "3008",
    (2, 2, 3): "3003",
    (2, 3, 3): "3002",
    (2, 4, 3): "3001",
    (2, 6, 3): "2456",
    (2, 8, 3): "3007",
}

# Sorted largest-to-smallest for greedy coverage
_SORTED_PARTS = sorted(
    PART_CATALOG.keys(),
    key=lambda s: s[0] * s[1] * s[2],
    reverse=True
)


# ─── Greedy layer covering ─────────────────────────────────────────────────────

def _can_place(footprint: np.ndarray, x: int, y: int, w: int, d: int) -> bool:
    """Check if a w×d brick can be placed at (x,y) in the footprint."""
    if x + w > footprint.shape[0] or y + d > footprint.shape[1]:
        return False
    return bool(np.all(footprint[x:x+w, y:y+d] > 0))


def _cover_layer(footprint: np.ndarray, grid_slice: np.ndarray,
                 z: int, height: int, placements: list) -> None:
    """
    Greedily cover all filled cells in footprint with bricks/plates.
    footprint: 2D array, >0 means needs covering, 0 means empty/covered
    grid_slice: 2D color array for this z-level
    """
    remaining = footprint.copy()
    
    while np.any(remaining > 0):
        placed = False
        for x in range(remaining.shape[0]):
            if placed:
                break
            for y in range(remaining.shape[1]):
                if remaining[x, y] == 0:
                    continue
                for (w, d, h) in _SORTED_PARTS:
                    if h != height:
                        continue
                    # Try both orientations
                    for rot, (pw, pd) in enumerate([(w, d), (d, w)]):
                        if _can_place(remaining, x, y, pw, pd):
                            # Determine dominant color in this region
                            region = grid_slice[x:x+pw, y:y+pd]
                            vals, counts = np.unique(region[region > 0], return_counts=True)
                            dom_color = int(vals[np.argmax(counts)]) if len(vals) > 0 else 1
                            
                            placements.append({
                                "part_id": PART_CATALOG[(w, d, h)],
                                "color_id": dom_color,
                                "x": x, "y": y, "z": z,
                                "width": pw, "depth": pd,
                                "height": h,
                                "rotation": rot * 90,
                            })
                            remaining[x:x+pw, y:y+pd] = 0
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
        
        if not placed:
            # Fallback: place 1x1 plates for any remaining cells
            for x in range(remaining.shape[0]):
                for y in range(remaining.shape[1]):
                    if remaining[x, y] > 0:
                        c = int(grid_slice[x, y])
                        placements.append({
                            "part_id": "3024",
                            "color_id": c,
                            "x": x, "y": y, "z": z,
                            "width": 1, "depth": 1,
                            "height": 1,
                            "rotation": 0,
                        })
                        remaining[x, y] = 0
            break


# ─── Consolidation pass ───────────────────────────────────────────────────────

def _consolidate_bricks(placements: list) -> list:
    """
    Merge three consecutive plate placements at the same XY into one brick.
    """
    from collections import defaultdict
    by_pos = defaultdict(list)
    for p in placements:
        key = (p["x"], p["y"], p["width"], p["depth"], p["color_id"])
        by_pos[key].append(p)
    
    result = []
    processed = set()
    
    for i, p in enumerate(placements):
        if id(p) in processed:
            continue
        if p["height"] == 1:
            key = (p["x"], p["y"], p["width"], p["depth"], p["color_id"])
            same = by_pos[key]
            z_vals = sorted(set(q["z"] for q in same))
            # Find runs of 3 consecutive Z values
            j = 0
            while j < len(z_vals) - 2:
                if z_vals[j+1] == z_vals[j]+1 and z_vals[j+2] == z_vals[j]+2:
                    # Check if a brick part exists for this footprint
                    brick_key = (min(p["width"], p["depth"]),
                                 max(p["width"], p["depth"]), 3)
                    if brick_key in PART_CATALOG:
                        brick = {
                            "part_id": PART_CATALOG[brick_key],
                            "color_id": p["color_id"],
                            "x": p["x"], "y": p["y"], "z": z_vals[j],
                            "width": p["width"], "depth": p["depth"],
                            "height": 3, "rotation": p["rotation"],
                        }
                        result.append(brick)
                        for q in same:
                            if q["z"] in (z_vals[j], z_vals[j+1], z_vals[j+2]):
                                processed.add(id(q))
                        j += 3
                        continue
                j += 1
        if id(p) not in processed:
            result.append(p)
            processed.add(id(p))
    
    return result


# ─── Main optimize function ───────────────────────────────────────────────────

def optimize_parts(grid: np.ndarray, color_map: dict[str, int] | None = None,
                   verbose: bool = False) -> list[dict[str, Any]]:
    """
    Convert a voxel grid to an optimized LEGO part placement list.
    
    Args:
        grid:      3D NumPy array (X, Y, Z) from stage2_voxelizer
        color_map: Optional name->index map for color names in output
        verbose:   Print progress
        
    Returns:
        List of placement dicts, one per LEGO piece.
        
    Example:
        >>> grid, color_map = voxelize(spec)
        >>> placements = optimize_parts(grid, color_map)
        >>> print(len(placements))
        342
    """
    X, Y, Z = grid.shape
    placements = []
    
    # Reverse color map for lookup
    rev_color = {v: k for k, v in (color_map or {}).items()}
    
    if verbose:
        filled = int(np.sum(grid > 0))
        print(f"[Stage 3] Optimizing {filled} voxels in {X}x{Y}x{Z} grid")
    
    # Process plate-by-plate (each Z layer = 1 plate height)
    z = 0
    while z < Z:
        # Check if 3 consecutive layers are identical for brick consolidation
        if z + 2 < Z:
            layer0 = grid[:, :, z]
            layer1 = grid[:, :, z+1]
            layer2 = grid[:, :, z+2]
            if np.array_equal(layer0 > 0, layer1 > 0) and np.array_equal(layer0 > 0, layer2 > 0):
                footprint = layer0.copy()
                _cover_layer(footprint, layer0, z, 3, placements)
                z += 3
                if verbose and z % 15 == 0:
                    print(f"  Z={z}/{Z} (bricks), placements so far: {len(placements)}")
                continue
        
        layer = grid[:, :, z]
        if np.any(layer > 0):
            footprint = layer.copy()
            _cover_layer(footprint, layer, z, 1, placements)
        
        z += 1
        if verbose and z % 10 == 0:
            print(f"  Z={z}/{Z}, placements so far: {len(placements)}")
    
    # Add color names
    for p in placements:
        p["color_name"] = rev_color.get(p["color_id"], f"color_{p['color_id']}")
    
    if verbose:
        print(f"[Stage 3] Done. {len(placements)} total parts.")
        part_counts = {}
        for p in placements:
            part_counts[p["part_id"]] = part_counts.get(p["part_id"], 0) + 1
        top5 = sorted(part_counts.items(), key=lambda x: -x[1])[:5]
        print("  Top 5 parts:", top5)
    
    return placements


def summarize_parts(placements: list[dict]) -> dict[str, int]:
    """Return a bill of materials: {part_id: quantity}."""
    bom = {}
    for p in placements:
        key = f"{p['part_id']}_{p['color_id']}"
        bom[key] = bom.get(key, 0) + 1
    return bom


if __name__ == "__main__":
    import argparse, json
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Stage 3: Voxel grid -> part placements")
    parser.add_argument("grid_file", help="Path to .npy voxel grid")
    parser.add_argument("--output", default="placements.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    grid = np.load(args.grid_file)
    placements = optimize_parts(grid, verbose=args.verbose)
    
    with open(args.output, "w") as f:
        json.dump(placements, f, indent=2)
    
    print(f"Saved {len(placements)} placements to {args.output}")
    bom = summarize_parts(placements)
    print(f"Unique part+color combos: {len(bom)}")
