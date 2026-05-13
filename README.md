# 🧱 text-to-lego

> **Type a name. Get a LEGO instruction booklet.**

An AI pipeline that converts a text prompt (e.g. `"Eiffel Tower"`) into:
- ✅ A list of real, buyable LEGO parts with quantities and colors
- ✅ A step-by-step `.ldr` instruction file (opens in BrickLink Studio)
- ✅ A PDF instruction booklet with a Bill of Materials
- ✅ Total build cost + suggested retail price if you sell it as a kit

---

## Pipeline Overview

```
Text Input
    │
    ▼
Stage 1: LLM Geometry Decomposer (stage1_decomposer.py)
    Multi-pass Claude/GPT-4o chain → structured geometry JSON
    │
    ▼
Stage 2: Voxelizer (stage2_voxelizer.py)
    Geometry primitives → 3D NumPy voxel grid
    Shapes: box, cylinder, cone, arch, strut, lattice
    │
    ▼
Stage 3: Part Optimizer (stage3_optimizer.py)
    Voxel grid → LEGO part placements
    Greedy layer-by-layer covering algorithm
    Brick consolidation + stagger enforcement
    │
    ▼
Stage 4: LDraw Writer (stage4_ldraw.py)
    Part placements → .ldr file
    Standard LDraw format (opens in BrickLink Studio, LDCad)
    Step-by-step instruction manifest
    │
    ▼
Stage 5: Pricing + PDF (stage5_pricing.py)
    BrickLink/default catalog pricing
    PDF instruction booklet (ReportLab)
    Retail price suggestions
```

---

## Quick Start

### 1. Install
```bash
git clone https://github.com/Monzingo89/text-to-lego.git
cd text-to-lego
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### 3. Get LEGO parts database (one-time)
```bash
python scripts/refresh_rebrickable_data.py
```

### 4. Build something
```bash
# Basic usage
python -m pipeline.runner "Eiffel Tower"

# More options
python -m pipeline.runner "a simple red house" --model anthropic --verbose
python -m pipeline.runner "Sydney Opera House" --output-dir ./builds --skip-pdf
```

Output files land in `output/<subject_name>/`:
- `geometry_spec.json` — LLM geometry decomposition
- `voxel_grid.npy` — 3D voxel array
- `placements.json` — LEGO part placement list
- `<name>.ldr` — **Open this in BrickLink Studio**
- `<name>_instructions.pdf` — Print-ready instruction booklet
- `build_manifest.json` — Full run summary

---

## Project Structure

```
text-to-lego/
├── pipeline/
│   ├── __init__.py
│   ├── runner.py              # Main entry point
│   ├── stage1_decomposer.py   # LLM → geometry JSON
│   ├── stage2_voxelizer.py    # Geometry → 3D voxels
│   ├── stage3_optimizer.py    # Voxels → LEGO parts
│   ├── stage4_ldraw.py        # Parts → .ldr file
│   └── stage5_pricing.py      # Parts → PDF + pricing
├── scripts/
│   └── refresh_rebrickable_data.py  # Download LEGO catalog CSVs
├── tests/
│   ├── __init__.py
│   └── test_voxelizer.py
├── data/                      # gitignored — created at runtime
│   ├── rebrickable/           # Downloaded CSVs
│   └── lego.db                # SQLite parts database
├── output/                    # gitignored — generated builds
├── .env.example               # Copy to .env and add API keys
├── requirements.txt
└── README.md
```

---

## Stage Details

### Stage 1 — Text → Geometry (LLM)
Uses a 3-pass chain-of-thought prompting strategy:
- **Pass 1**: Research the geometry of the subject (proportions, structural components)
- **Pass 2**: Determine ideal LEGO scale (target stud dimensions)
- **Pass 3**: Convert to voxel primitive spec (JSON schema)

Works with Claude (Anthropic) or GPT-4o (OpenAI). Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

### Stage 2 — Geometry → Voxels
Converts geometric primitives to a 3D NumPy array. Supported shapes:
- `box` — rectangular fill
- `cylinder` — circular cross-section (uses ellipse equation per Z layer)
- `cone` — tapering cylinder
- `arch` — box with cylindrical void
- `strut` — Bresenham 3D diagonal line with configurable thickness
- `lattice` — diagonal cross-bracing pattern (for Eiffel Tower legs etc.)

Coordinate system: X=studs left-right, Y=studs front-back, Z=plates bottom-top.

### Stage 3 — Voxels → Parts
Greedy layer-by-layer covering algorithm:
1. Process each Z-plate layer
2. Detect if 3 consecutive layers are identical → use bricks instead of plates
3. For each layer, find largest valid LEGO part that fits unfilled voxels
4. Color assignment: most common color in the brick's covered voxel region

Part catalog covers 25 standard bricks and plates (1×1 to 4×8).

### Stage 4 — Parts → LDraw
Writes a standard `.ldr` file readable by any LDraw-compatible software.
Includes `STEP` markers for instruction sequence. Also writes a step manifest JSON.

**Coordinate conversion**: LEGO studs → LDraw units (1 stud = 20 LDU, 1 plate = 8 LDU).

### Stage 5 — Pricing + PDF
Aggregates parts into a Bill of Materials, looks up prices (BrickLink API or default
catalog), and generates a ReportLab PDF with cover page, BOM table, and instructions.

Retail pricing formula: `standard = (parts_cost × 2.5) + $15.00`.

---

## Development Milestones

- [x] **M1**: Repo structure + Rebrickable data loader
- [x] **M2**: Stage 2 voxelizer (box, cylinder, cone, strut, lattice)
- [x] **M3**: Stage 3 greedy part optimizer
- [x] **M4**: Stage 4 LDraw file writer
- [x] **M5**: Stage 1 LLM decomposer (3-pass chain)
- [x] **M6**: Stage 5 pricing + PDF
- [x] **M7**: Full pipeline runner
- [ ] **M8**: End-to-end Eiffel Tower test
- [ ] **M9**: Web UI (FastAPI + three.js LDraw viewer)
- [ ] **M10**: Fine-tuned model on Rebrickable set inventory data

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | One of these | Claude API key for Stage 1 |
| `OPENAI_API_KEY` | One of these | GPT-4o key for Stage 1 |
| `REBRICKABLE_API_KEY` | Optional | For API queries (bulk data uses free CSVs) |
| `BRICKLINK_CONSUMER_KEY` | Optional | For live BrickLink pricing |
| `BRICKLINK_CONSUMER_SECRET` | Optional | BrickLink OAuth |
| `BRICKLINK_ACCESS_TOKEN` | Optional | BrickLink OAuth |
| `BRICKLINK_ACCESS_TOKEN_SECRET` | Optional | BrickLink OAuth |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Data Sources

- **Parts/Colors**: [Rebrickable](https://rebrickable.com/downloads/) — free daily CSV exports, no API key needed
- **Pricing**: [BrickLink API](https://www.bricklink.com/v3/api.page) — requires OAuth + static IP registration
- **LDraw format**: [ldraw.org](https://ldraw.org) — open standard for LEGO CAD files
- **Viewer**: [BrickLink Studio](https://www.bricklink.com/v3/studio/download.page) — free, opens `.ldr` files

---

## License

MIT — see [LICENSE](LICENSE)

> LEGO® is a trademark of the LEGO Group. This project is not affiliated with or endorsed by the LEGO Group.
