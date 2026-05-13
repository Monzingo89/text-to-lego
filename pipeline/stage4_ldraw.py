"""
stage4_ldraw.py
===============
Stage 4: Part Placements -> LDraw .ldr File + Instruction Steps

Converts the list of LEGO part placements from Stage 3 into:
  1. An LDraw .ldr file that can be opened in BrickLink Studio, LDCad, etc.
  2. Step-by-step instruction data (JSON list of steps with parts per step)
  3. (Optional) Render calls to LDView/Studio CLI for instruction images

LDraw file format reference:
  Line type 1 (part): 1 <color> <x> <y> <z> <rotation 3x3 matrix> <partfile>
  Coordinates in LDraw units (1 LDraw unit = 0.4mm)
  1 stud = 20 LDraw units
  1 plate = 8 LDraw units tall
  1 brick = 24 LDraw units tall

LDraw color reference (common):
  0=Black, 1=Blue, 2=Green, 4=Red, 7=Light Gray, 14=Yellow, 15=White, 71=Light Bluish Gray
"""

import os
import json
import math
from typing import Any
from pathlib import Path


# ─── Unit conversions ─────────────────────────────────────────────────────────

STUD_TO_LDU = 20      # 1 stud = 20 LDraw units (horizontal)
PLATE_TO_LDU = 8      # 1 plate = 8 LDraw units (vertical)
BRICK_TO_LDU = 24     # 1 brick = 24 LDraw units (3 plates)


# ─── Color mapping (Rebrickable color_id -> LDraw color code) ─────────────────
# Rebrickable IDs: https://rebrickable.com/api/v3/lego/colors/
# LDraw colors: https://www.ldraw.org/article/547.html

REBRICKABLE_TO_LDRAW: dict[int, int] = {
    0:   15,   # White
    1:   71,   # Light Bluish Gray
    2:    7,   # Light Gray
    3:   72,   # Dark Bluish Gray
    4:   11,   # Black  
    5:    4,   # Red
    6:  320,   # Dark Red
    7:  484,   # Dark Orange
    8:   25,   # Orange
    9:  366,   # Bright Light Orange
    10:  14,   # Yellow
    11: 226,   # Bright Light Yellow
    12:  10,   # Bright Green / Lime
    13:   2,   # Green
    14:   6,   # Dark Green
    15:   1,   # Blue
    16:  73,   # Medium Blue
    17: 212,   # Bright Light Blue
    18:   9,   # Light Purple / Lavender
    19:  26,   # Black (placeholder)
}


def rebrickable_to_ldraw_color(color_id: int) -> int:
    """Map Rebrickable color ID to LDraw color code."""
    return REBRICKABLE_TO_LDRAW.get(color_id, 7)  # default Light Gray


# ─── Rotation matrices ────────────────────────────────────────────────────────

def rotation_matrix_str(rotation_deg: int) -> str:
    """Return the 9-value LDraw rotation matrix string for a given Y-axis rotation."""
    if rotation_deg == 0:
        return "1 0 0 0 1 0 0 0 1"
    elif rotation_deg == 90:
        return "0 0 1 0 1 0 -1 0 0"
    elif rotation_deg == 180:
        return "-1 0 0 0 1 0 0 0 -1"
    elif rotation_deg == 270:
        return "0 0 -1 0 1 0 1 0 0"
    else:
        rad = math.radians(rotation_deg)
        c, s = round(math.cos(rad), 6), round(math.sin(rad), 6)
        return f"{c} 0 {s} 0 1 0 {-s} 0 {c}"


# ─── LDraw line builders ──────────────────────────────────────────────────────

def placement_to_ldraw_line(p: dict[str, Any]) -> str:
    """
    Convert a single placement dict to an LDraw type-1 line.
    
    LDraw coordinate system:
      X = right, Y = DOWN (inverted!), Z = back
    We map: stud_x -> LDU_x, plate_z -> LDU_y (negated, LDraw Y is down),
             stud_y -> LDU_z
    """
    ldraw_x = p["x"] * STUD_TO_LDU
    ldraw_y = -(p["z"] * PLATE_TO_LDU)  # LDraw Y is inverted (positive = down)
    ldraw_z = p["y"] * STUD_TO_LDU
    
    ldraw_color = rebrickable_to_ldraw_color(p["color_id"])
    rotation = rotation_matrix_str(p.get("rotation", 0))
    part_file = f"{p['part_id']}.dat"
    
    return f"1 {ldraw_color} {ldraw_x} {ldraw_y} {ldraw_z} {rotation} {part_file}"


def make_step_comment(step_num: int, total: int) -> str:
    return f"0 STEP  // Step {step_num} of {total}"


# ─── Instruction step splitter ────────────────────────────────────────────────

def split_into_steps(placements: list[dict], max_parts_per_step: int = 20) -> list[list[dict]]:
    """
    Split placements into build steps. Strategy: group by Z-layer.
    Each Z layer that has <= max_parts_per_step parts becomes one step.
    Larger layers are split into sub-steps.
    """
    from collections import defaultdict
    by_z = defaultdict(list)
    for p in placements:
        by_z[p["z"]].append(p)
    
    steps = []
    for z in sorted(by_z.keys()):
        layer_parts = by_z[z]
        if len(layer_parts) <= max_parts_per_step:
            steps.append(layer_parts)
        else:
            # Split into chunks
            for i in range(0, len(layer_parts), max_parts_per_step):
                steps.append(layer_parts[i:i + max_parts_per_step])
    
    return steps


# ─── LDraw file writer ────────────────────────────────────────────────────────

def write_ldraw(
    placements: list[dict[str, Any]],
    output_path: str,
    title: str = "Text-to-LEGO Model",
    author: str = "Monzingo89/text-to-lego",
    include_steps: bool = True,
    max_parts_per_step: int = 20,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Write LEGO placements to an LDraw .ldr file.
    
    Args:
        placements:          List of placement dicts from stage3_optimizer
        output_path:         Output .ldr file path
        title:               Model title (shown in LDraw viewers)
        author:              Author string
        include_steps:       Add STEP comments for instruction sequence
        max_parts_per_step:  Maximum pieces per instruction step
        verbose:             Print progress
        
    Returns:
        Dict with metadata: steps, total_parts, output_path
        
    Example:
        >>> placements = optimize_parts(grid, color_map)
        >>> meta = write_ldraw(placements, "eiffel_tower.ldr", title="Eiffel Tower")
        >>> print(meta["steps"])
        24
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    steps = split_into_steps(placements, max_parts_per_step) if include_steps else [placements]
    
    if verbose:
        print(f"[Stage 4] Writing LDraw file: {output_path}")
        print(f"  Parts: {len(placements)}, Steps: {len(steps)}")
    
    lines = []
    
    # Header
    lines.append(f"0 {title}")
    lines.append(f"0 Name: {os.path.basename(output_path)}")
    lines.append(f"0 Author: {author}")
    lines.append("0 Unofficial Model")
    lines.append("0 ROTATION CENTER 0 0 0 1 "Custom"")
    lines.append("")
    
    # Parts by step
    for step_idx, step_parts in enumerate(steps, start=1):
        if include_steps:
            lines.append(f"0 STEP")
            lines.append(f"0 // Step {step_idx} of {len(steps)} — {len(step_parts)} parts")
        
        for p in step_parts:
            lines.append(placement_to_ldraw_line(p))
        
        if verbose and step_idx % 10 == 0:
            print(f"  Written step {step_idx}/{len(steps)}")
    
    lines.append("")
    lines.append("0 // End of model - generated by github.com/Monzingo89/text-to-lego")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    if verbose:
        print(f"[Stage 4] Wrote {len(lines)} lines to {output_path}")
    
    # Also write step manifest JSON
    step_manifest = []
    for i, step_parts in enumerate(steps, 1):
        step_manifest.append({
            "step": i,
            "parts": len(step_parts),
            "z_range": [min(p["z"] for p in step_parts), max(p["z"] for p in step_parts)],
        })
    
    manifest_path = output_path.replace(".ldr", "_steps.json")
    with open(manifest_path, "w") as f:
        json.dump(step_manifest, f, indent=2)
    
    return {
        "output_path": output_path,
        "manifest_path": manifest_path,
        "total_parts": len(placements),
        "steps": len(steps),
    }


# ─── Bill of Materials formatter ──────────────────────────────────────────────

def format_bom_table(placements: list[dict]) -> str:
    """Return a formatted text table of the bill of materials."""
    from collections import defaultdict
    counts = defaultdict(int)
    names = {}
    
    for p in placements:
        key = (p["part_id"], p["color_id"])
        counts[key] += 1
        names[key] = (p.get("color_name", str(p["color_id"])),)
    
    lines = ["Part ID    | Color              | Qty"]
    lines.append("-" * 40)
    for (part_id, color_id), qty in sorted(counts.items(), key=lambda x: -x[1]):
        color_name = names[(part_id, color_id)][0]
        lines.append(f"{part_id:<10} | {color_name:<18} | {qty}")
    
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Part placements -> LDraw file")
    parser.add_argument("placements_file", help="JSON file from stage3_optimizer")
    parser.add_argument("--output", default="output/model.ldr")
    parser.add_argument("--title", default="Text-to-LEGO Model")
    parser.add_argument("--no-steps", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    with open(args.placements_file) as f:
        placements = json.load(f)
    
    meta = write_ldraw(
        placements, args.output,
        title=args.title,
        include_steps=not args.no_steps,
        verbose=args.verbose,
    )
    
    print(f"LDraw file: {meta['output_path']}")
    print(f"Steps: {meta['steps']}")
    print(f"Total parts: {meta['total_parts']}")
    print()
    print(format_bom_table(placements))
