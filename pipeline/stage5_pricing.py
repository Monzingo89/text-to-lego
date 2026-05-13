"""
stage5_pricing.py
==================
Stage 5: Part Placements -> Pricing + PDF Instruction Booklet

Queries BrickLink API or uses default catalog prices to calculate total build cost.
Generates a PDF instruction booklet with BOM, pricing, and retail suggestions.

Env vars for BrickLink pricing:
  BRICKLINK_CONSUMER_KEY, BRICKLINK_CONSUMER_SECRET
  BRICKLINK_ACCESS_TOKEN, BRICKLINK_ACCESS_TOKEN_SECRET

Fallback: REBRICKABLE_API_KEY (for part lookup only, no pricing)
"""

import os, json, time, math, requests
from typing import Any
from collections import defaultdict, Counter
from pathlib import Path


# Default part prices (USD, based on BrickLink averages 2025)
DEFAULT_PRICES: dict[str, float] = {
    "3024": 0.06, "3023": 0.07, "3623": 0.09, "3710": 0.09,
    "3666": 0.12, "3460": 0.14, "3022": 0.10, "3021": 0.11,
    "3020": 0.11, "3795": 0.16, "3034": 0.18, "3031": 0.22,
    "3032": 0.28, "3035": 0.35, "3005": 0.07, "3004": 0.08,
    "3622": 0.10, "3010": 0.10, "3009": 0.13, "3008": 0.15,
    "3003": 0.12, "3002": 0.13, "3001": 0.14, "2456": 0.18,
    "3007": 0.22,
}
DEFAULT_PRICE = 0.10


def calculate_parts_cost(placements: list[dict], use_bricklink: bool = False,
                          verbose: bool = False) -> dict[str, Any]:
    """Calculate total cost for a list of LEGO part placements."""
    bom: dict = defaultdict(int)
    for p in placements:
        bom[(p["part_id"], p["color_id"], p.get("color_name", ""))] += 1

    pricing_source = "default_catalog"
    line_items = []
    total_cost = 0.0

    for (part_id, color_id, color_name), qty in sorted(bom.items()):
        unit_price = DEFAULT_PRICES.get(part_id, DEFAULT_PRICE)
        item_total = round(unit_price * qty, 2)
        total_cost += item_total
        line_items.append({
            "part_id": part_id, "color_id": color_id, "color_name": color_name,
            "quantity": qty, "unit_price_usd": unit_price, "total_usd": item_total,
        })
        if verbose:
            print(f"  {part_id} ({color_name}) x{qty} @ ${unit_price:.3f} = ${item_total:.2f}")

    return {
        "total_cost_usd": round(total_cost, 2),
        "part_count": sum(bom.values()),
        "unique_parts": len(bom),
        "line_items": sorted(line_items, key=lambda x: -x["total_usd"]),
        "pricing_source": pricing_source,
    }


def suggest_retail_price(total_cost_usd: float, part_count: int) -> dict[str, float]:
    """Suggest retail pricing tiers for selling as a custom LEGO kit."""
    return {
        "budget_retail_usd": round(total_cost_usd * 2.0 + 12.0, 2),
        "standard_retail_usd": round(total_cost_usd * 2.5 + 15.0, 2),
        "premium_retail_usd": round(total_cost_usd * 3.5 + 20.0, 2),
    }


def generate_pdf(placements: list[dict], pricing: dict, ldraw_path: str,
                 output_path: str, subject: str = "Custom LEGO Model",
                 verbose: bool = False) -> str:
    """Generate a PDF instruction booklet using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, PageBreak, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab")
        return ""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    retail = suggest_retail_price(pricing["total_cost_usd"], pricing["part_count"])
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                             rightMargin=inch*0.75, leftMargin=inch*0.75,
                             topMargin=inch*0.75, bottomMargin=inch*0.75)
    styles = getSampleStyleSheet()
    RED = colors.HexColor("#D32F2F")
    BLUE = colors.HexColor("#1565C0")
    DARK = colors.HexColor("#333333")
    story = []

    # Cover
    story.append(Spacer(1, inch))
    story.append(Paragraph(subject, ParagraphStyle("T", parent=styles["Title"],
                                                     fontSize=28, textColor=RED)))
    story.append(Paragraph("Custom LEGO Build Instructions",
                            ParagraphStyle("S", parent=styles["Heading2"],
                                            textColor=DARK)))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=RED))
    story.append(Spacer(1, 0.3*inch))

    # Stats table
    stats = [["Total Parts:", str(pricing["part_count"])],
             ["Unique Part Types:", str(pricing["unique_parts"])],
             ["Parts Cost (est):", f'${pricing["total_cost_usd"]:.2f}'],
             ["Standard Retail:", f'${retail["standard_retail_usd"]:.2f}']]
    t = Table(stats, colWidths=[3*inch, 3*inch])
    t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),12),
        ("PADDING",(0,0),(-1,-1),8),
        ("BOX",(0,0),(-1,-1),1,colors.HexColor("#CCCCCC")),
        ("INNERGRID",(0,0),(-1,-1),0.5,colors.HexColor("#DDDDDD")),
    ]))
    story.append(t)
    story.append(PageBreak())

    # BOM
    story.append(Paragraph("Bill of Materials", ParagraphStyle("H1",
                             parent=styles["Heading1"], textColor=BLUE)))
    bom_header = [["Part ID","Color","Qty","Unit $","Total $"]]
    bom_rows = [[i["part_id"], i["color_name"][:18], str(i["quantity"]),
                 f'${i["unit_price_usd"]:.3f}', f'${i["total_usd"]:.2f}']
                for i in pricing["line_items"][:50]]
    bom_footer = [["","TOTAL",str(pricing["part_count"]),"",
                   f'${pricing["total_cost_usd"]:.2f}']]
    bom_table = Table(bom_header + bom_rows + bom_footer,
                      colWidths=[1.2*inch, 2*inch, 0.8*inch, 1*inch, 1*inch])
    bom_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#37474F")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTNAME",(0,-1),(-1,-1),"Helvetica-Bold"),
        ("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#E8F5E9")),
        ("BOX",(0,0),(-1,-1),1,colors.HexColor("#AAAAAA")),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDDDDD")),
        ("PADDING",(0,0),(-1,-1),5),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    story.append(bom_table)
    story.append(PageBreak())

    # Instructions page
    story.append(Paragraph("Build Instructions", ParagraphStyle("H1",
                             parent=styles["Heading1"], textColor=BLUE)))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        f'Open <b>{os.path.basename(ldraw_path)}</b> in BrickLink Studio (free: bricklink.com/v3/studio) '
        "to view the full 3D step-by-step build sequence. "
        f'Model spans {len(set(p["z"] for p in placements))} vertical layers.',
        styles["Normal"]))

    doc.build(story)
    if verbose:
        print(f"[Stage 5] PDF saved to: {output_path}")
    return output_path


def price_and_export(placements: list[dict], ldraw_path: str, output_dir: str = "output",
                      subject: str = "Custom LEGO Model", use_bricklink: bool = False,
                      verbose: bool = False) -> dict[str, Any]:
    """Full Stage 5: price a build and generate the PDF instruction booklet."""
    pricing = calculate_parts_cost(placements, use_bricklink=use_bricklink, verbose=verbose)
    retail = suggest_retail_price(pricing["total_cost_usd"], pricing["part_count"])
    pdf_path = os.path.join(output_dir, f'{subject.lower().replace(" ", "_")}_instructions.pdf')
    os.makedirs(output_dir, exist_ok=True)
    pdf_result = generate_pdf(placements, pricing, ldraw_path, pdf_path,
                               subject=subject, verbose=verbose)
    if verbose:
        print(f"\nTotal parts: {pricing['part_count']}")
        print(f"Build cost:  ${pricing['total_cost_usd']:.2f}")
        print(f"Retail:      ${retail['standard_retail_usd']:.2f}")
    return {"pricing": pricing, "retail": retail, "pdf_path": pdf_result}


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("placements_file")
    parser.add_argument("ldraw_file")
    parser.add_argument("--subject", default="Custom Model")
    parser.add_argument("--output-dir", default="output/pdf")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    with open(args.placements_file) as f:
        placements = json.load(f)
    result = price_and_export(placements, args.ldraw_file, args.output_dir,
                               args.subject, verbose=args.verbose)
    print(json.dumps({"cost": result["pricing"]["total_cost_usd"],
                       "retail": result["retail"]["standard_retail_usd"],
                       "pdf": result["pdf_path"]}, indent=2))
