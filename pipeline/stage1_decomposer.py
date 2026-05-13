"""
stage1_decomposer.py
====================
Stage 1: Text → Structured 3D Geometry Description

Uses a multi-pass LLM chain to convert a natural language prompt (e.g. "Eiffel Tower")
into a structured JSON schema describing the model as geometric primitives at LEGO scale.

Three-pass strategy:
  Pass 1 - Research:   Deep geometric description of the subject
  Pass 2 - Scale:      Determine ideal LEGO scale and target stud dimensions
  Pass 3 - Voxelspec:  Convert to a list of primitive voxel-grid specs

Output schema:
{
  "subject": str,
  "scale_factor": float,           # e.g. 1/300 for Eiffel Tower
  "target_studs": [x, y, z],       # bounding box in studs
  "color_palette": {name: rgb},    # named colors used in this model
  "primitives": [
    {
      "name": str,                 # e.g. "left_leg"
      "shape": str,                # box | cylinder | cone | arch | strut | lattice
      "origin": [x, y, z],         # bottom-left-front corner in studs
      "size": [w, d, h],           # in studs
      "color": str,                # key from color_palette
      "fill_density": float,       # 0.0-1.0, for lattice structures
      "angle_deg": float | null    # for struts: angle from vertical
    }
  ]
}
"""

import json
import os
from typing import Any

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ─── Prompt templates ────────────────────────────────────────────────────────

_PASS1_SYSTEM = """You are an expert architect and 3D modeler. 
Your task is to describe the 3D geometry of a subject in precise, structured terms.
Focus on: overall shape, proportions, key structural elements, symmetry axes, 
approximate real-world dimensions, and which features are visually essential vs decorative.
Be concise and factual. Do not use vague language."""

_PASS1_USER = """Describe the 3D geometry of: {subject}

Return a JSON object with these fields:
- real_world_height_m: approximate real height in meters
- real_world_width_m: approximate real width at base in meters  
- overall_shape: one-sentence summary of the dominant form
- structural_components: list of objects, each with:
    - name: descriptive name
    - shape_type: box | cylinder | cone | arch | strut | irregular
    - relative_position: description of where it sits (e.g. "bottom 30 percent, four corners")
    - relative_size: fraction of total height/width
    - visual_importance: high | medium | low
- symmetry: rotational | bilateral | none
- notes: any special geometry considerations
"""

_PASS2_SYSTEM = """You are a LEGO master builder. 
Given real-world dimensions and a geometric description, determine the ideal LEGO build scale.
LEGO scale reference:
- 1 stud = 8mm horizontal
- 1 plate = 3.2mm tall (Z)  
- 1 brick = 9.6mm tall (3 plates)
- Minifigure scale: ~1:42
- Architecture/landmark scale: typically 1:150 to 1:300
- Small desktop model: 1:500+
"""

_PASS2_USER = """Given this geometric description:
{geo_description}

Determine the ideal LEGO build parameters for a desktop landmark model.
Return a JSON object with:
- recommended_scale: fraction (e.g. 0.00333 for 1:300)
- target_height_studs: integer
- target_width_studs: integer  
- target_depth_studs: integer
- total_estimated_bricks: rough estimate (100-5000)
- scale_rationale: one sentence explaining the choice
- simplification_notes: list of features to simplify at this scale
"""

_PASS3_SYSTEM = """You are a voxel 3D modeler converting geometric descriptions to LEGO-scale voxel primitives.
Each primitive defines a 3D shape to be filled with voxels.
Coordinate system: origin (0,0,0) is bottom-left-front of the model.
X = left-to-right (studs), Y = front-to-back (studs), Z = bottom-to-top (plates, 1/3 of a stud height).
All dimensions in LEGO units (studs for X/Y, plates for Z unless noted).
"""

_PASS3_USER = """Convert this scaled LEGO model description to voxel primitives:

Subject: {subject}
Target size: {target_studs_x} x {target_studs_y} x {target_studs_z} studs
Geometric components: {components_json}

Return a JSON object matching this schema exactly:
{{
  "subject": "{subject}",
  "scale_factor": <float>,
  "target_studs": [x, y, z],
  "color_palette": {{"primary": [R,G,B], "secondary": [R,G,B], "accent": [R,G,B]}},
  "primitives": [
    {{
      "name": "<component name>",
      "shape": "<box|cylinder|cone|arch|strut|lattice>",
      "origin": [x, y, z],
      "size": [width_studs, depth_studs, height_plates],
      "color": "<color name from palette>",
      "fill_density": <0.0-1.0>,
      "angle_deg": <float or null>
    }}
  ]
}}

Rules:
- Use plate units (1/3 of stud) for Z/height to allow fine vertical resolution
- Struts use angle_deg (degrees from vertical, 0=straight up)
- Lattice fill_density: 1.0=solid, 0.3=open truss, 0.1=very sparse
- All primitives must fit within target_studs bounding box
- Include at least one primitive per major structural component
"""


# ─── LLM call helper ─────────────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_prompt: str, model: str = "auto") -> str:
    """Call the configured LLM and return the response text."""
    
    if model == "auto":
        if _HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
            model = "anthropic"
        elif _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            model = "openai"
        else:
            raise RuntimeError(
                "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
            )
    
    if model == "anthropic":
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
    
    elif model == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unknown model: {model}. Use 'anthropic', 'openai', or 'auto'.")


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a response string."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response:\n{text[:500]}")
    return json.loads(text[start:end])


# ─── Main decompose function ──────────────────────────────────────────────────

def decompose_text(subject: str, model: str = "auto", verbose: bool = False) -> dict[str, Any]:
    """
    Convert a text prompt to a structured 3D geometry description at LEGO scale.
    
    Args:
        subject:  Natural language description, e.g. "Eiffel Tower" or "a simple house"
        model:    "anthropic", "openai", or "auto" (uses whichever API key is available)
        verbose:  If True, prints intermediate LLM responses
        
    Returns:
        Structured geometry dict matching the voxelspec schema.
        
    Example:
        >>> spec = decompose_text("Eiffel Tower")
        >>> print(spec["target_studs"])
        [32, 32, 96]
        >>> print(len(spec["primitives"]))
        8
    """
    if verbose:
        print(f"[Stage 1] Decomposing: {subject!r}")
    
    # ── Pass 1: Research geometry ──────────────────────────────────────────
    if verbose:
        print("  Pass 1: Researching geometry...")
    
    pass1_response = _call_llm(
        system_prompt=_PASS1_SYSTEM,
        user_prompt=_PASS1_USER.format(subject=subject),
        model=model,
    )
    
    if verbose:
        print(f"  Pass 1 response length: {len(pass1_response)} chars")
    
    pass1_data = _extract_json(pass1_response)
    
    # ── Pass 2: Determine scale ────────────────────────────────────────────
    if verbose:
        print("  Pass 2: Determining LEGO scale...")
    
    pass2_response = _call_llm(
        system_prompt=_PASS2_SYSTEM,
        user_prompt=_PASS2_USER.format(geo_description=json.dumps(pass1_data, indent=2)),
        model=model,
    )
    
    pass2_data = _extract_json(pass2_response)
    
    if verbose:
        print(f"  Scale: 1:{int(1/pass2_data['recommended_scale'])}")
        print(f"  Target: {pass2_data['target_width_studs']}x{pass2_data['target_depth_studs'] if 'target_depth_studs' in pass2_data else pass2_data['target_width_studs']}x{pass2_data['target_height_studs']} studs")
    
    # ── Pass 3: Generate voxel spec ────────────────────────────────────────
    if verbose:
        print("  Pass 3: Generating voxel primitives...")
    
    target_x = pass2_data.get("target_width_studs", 32)
    target_y = pass2_data.get("target_depth_studs", pass2_data.get("target_width_studs", 32))
    target_z = pass2_data.get("target_height_studs", 96)
    
    components = pass1_data.get("structural_components", [])
    
    pass3_response = _call_llm(
        system_prompt=_PASS3_SYSTEM,
        user_prompt=_PASS3_USER.format(
            subject=subject,
            target_studs_x=target_x,
            target_studs_y=target_y,
            target_studs_z=target_z * 3,  # convert to plates
            components_json=json.dumps(components, indent=2),
        ),
        model=model,
    )
    
    voxel_spec = _extract_json(pass3_response)
    
    # Validate and normalise
    if "subject" not in voxel_spec:
        voxel_spec["subject"] = subject
    if "target_studs" not in voxel_spec:
        voxel_spec["target_studs"] = [target_x, target_y, target_z * 3]
    if "primitives" not in voxel_spec or not voxel_spec["primitives"]:
        raise ValueError("LLM returned empty primitives list in Pass 3.")
    
    if verbose:
        print(f"  Generated {len(voxel_spec['primitives'])} primitives")
        for p in voxel_spec["primitives"]:
            print(f"    - {p['name']}: {p['shape']} @ {p['origin']} size {p['size']}")
    
    return voxel_spec


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1: Text → 3D geometry spec")
    parser.add_argument("subject", help="Subject to decompose, e.g. 'Eiffel Tower'")
    parser.add_argument("--model", default="auto", choices=["auto", "anthropic", "openai"])
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    from dotenv import load_dotenv
    load_dotenv()
    
    spec = decompose_text(args.subject, model=args.model, verbose=args.verbose)
    
    output_path = args.output or f"{args.subject.lower().replace(' ', '_')}_spec.json"
    with open(output_path, "w") as f:
        json.dump(spec, f, indent=2)
    
    print(f"\n✅ Geometry spec saved to: {output_path}")
    print(f"   Primitives: {len(spec['primitives'])}")
    print(f"   Target: {spec['target_studs']} studs")
