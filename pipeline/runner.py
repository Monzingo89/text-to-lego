"""
runner.py -- Main Pipeline Runner: Text -> Full LEGO Build Output

Usage:
    python -m pipeline.runner "Eiffel Tower"
    python -m pipeline.runner "a simple house" --verbose
    python -m pipeline.runner "Eiffel Tower" --output-dir ./my_builds --skip-pdf
"""

import os, sys, json, time, argparse
from dotenv import load_dotenv
load_dotenv()


def run_pipeline(subject, output_dir="output", model="auto",
                  verbose=False, skip_pdf=False):
    """Run the complete text-to-LEGO pipeline for a given subject."""
    safe = subject.lower().replace(" ","_").replace("/","_")
    bdir = os.path.join(output_dir, safe)
    os.makedirs(bdir, exist_ok=True)
    print(f"\n=== TEXT-TO-LEGO: {subject!r} ===")
    print(f"    Output: {bdir}\n")
    t0 = time.time()

    # Stage 1
    print("[1/5] Decomposing text...")
    from pipeline.stage1_decomposer import decompose_text
    spec = decompose_text(subject, model=model, verbose=verbose)
    sp = os.path.join(bdir,"geometry_spec.json")
    with open(sp,"w") as f: json.dump(spec,f,indent=2)
    print(f"  {len(spec['primitives'])} primitives, target {spec['target_studs']}")

    # Stage 2
    print("[2/5] Voxelizing...")
    import numpy as np
    from pipeline.stage2_voxelizer import voxelize, save_grid
    grid, color_map = voxelize(spec, verbose=verbose)
    gp = os.path.join(bdir,"voxel_grid.npy")
    save_grid(grid, gp)
    filled = int(np.sum(grid>0))
    print(f"  Grid {grid.shape}, {filled} filled voxels")

    # Stage 3
    print("[3/5] Optimizing parts...")
    from pipeline.stage3_optimizer import optimize_parts, summarize_parts
    placements = optimize_parts(grid, color_map=color_map, verbose=verbose)
    pp = os.path.join(bdir,"placements.json")
    with open(pp,"w") as f: json.dump(placements,f,indent=2)
    bom = summarize_parts(placements)
    print(f"  {len(placements)} parts, {len(bom)} unique")

    # Stage 4
    print("[4/5] Writing LDraw file...")
    from pipeline.stage4_ldraw import write_ldraw
    ldr = os.path.join(bdir, f"{safe}.ldr")
    meta = write_ldraw(placements, ldr, title=subject, verbose=verbose)
    print(f"  {ldr} ({meta['steps']} steps)")

    # Stage 5
    pdf_path, pricing = None, None
    if not skip_pdf:
        print("[5/5] Pricing + PDF...")
        from pipeline.stage5_pricing import price_and_export
        r5 = price_and_export(placements, ldr, output_dir=bdir,
                               subject=subject, verbose=verbose)
        pricing, pdf_path = r5["pricing"], r5["pdf_path"]
        print(f"  Cost ${pricing['total_cost_usd']:.2f}, retail ${r5['retail']['standard_retail_usd']:.2f}")
    else:
        print("[5/5] Skipped")

    elapsed = round(time.time()-t0, 2)
    print(f"\nDone in {elapsed}s. Open {ldr} in BrickLink Studio.")

    result = {
        "subject": subject, "build_dir": bdir,
        "spec_path": sp, "grid_path": gp, "placements_path": pp,
        "ldraw_path": ldr, "pdf_path": pdf_path,
        "total_parts": len(placements), "unique_parts": len(bom),
        "total_time_s": elapsed,
    }
    if pricing:
        result["cost_usd"] = pricing["total_cost_usd"]
    manifest = os.path.join(bdir,"build_manifest.json")
    with open(manifest,"w") as f: json.dump(result,f,indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description="Text-to-LEGO pipeline")
    p.add_argument("subject")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--model", default="auto",
                   choices=["auto","anthropic","openai"])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-pdf", action="store_true")
    args = p.parse_args()
    result = run_pipeline(args.subject, args.output_dir,
                           args.model, args.verbose, args.skip_pdf)
    keys = ["total_parts","unique_parts","cost_usd","total_time_s","ldraw_path","pdf_path"]
    print(json.dumps({k:v for k,v in result.items() if k in keys}, indent=2))


if __name__ == "__main__":
    main()
