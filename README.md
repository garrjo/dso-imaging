# DSO Imaging Framework
**Drag-Scale-Object (DSO)** geometric image recovery for TEM and other microscopy.

Uses known reference structures (e.g., graphene lattice) to measure and correct local distortion and blur.

## Theory
The DSO framework treats image degradation as a measurable geometric problem:

# DSO Imaging Framework

**Drag-Scale-Object (DSO)** geometric image recovery for TEM and other microscopy.

Uses known reference structures (e.g., graphene lattice) to measure and correct local distortion and blur.

## Theory

The DSO framework treats image degradation as a measurable geometric problem:
```
αΩ = D · Ω · O
```

Where:
- **Ω** (Scale) - Reference geometry at measurement scale
- **D** (Drag) - Local displacement and blur field
- **O** (Object) - The recovered structure

## API

| Function | Name | Purpose |
|----------|------|---------|
| `PSC` | P-Scale Coefficient | Generate reference geometry |
| `WDC` | W-Drag Coefficient | Measure displacement + blur |
| `GCO` | G-Corrected Output | Apply correction |

## Installation
```bash
pip install numpy scipy matplotlib pillow
```

Download `dso_framework.py` and import directly.

## Usage

### Quick Start
```python
from dso_framework import dso_recover

result = dso_recover(image, pixel_size_nm=0.02, material='graphene')
corrected = result['output']
```

### Step by Step
```python
from dso_framework import PSC_graphene, WDC_measure, GCO

# PSC: Establish reference geometry
positions, meta = PSC_graphene(image.shape, pixel_size_nm=0.02)

# WDC: Measure local distortion and blur
wdc = WDC_measure(image, positions, dark_atoms=True)

# GCO: Generate corrected output
gco = GCO(image, wdc, mode='auto')
corrected = gco['output']
```

### Modes

| Mode | Use When | Pipeline |
|------|----------|----------|
| `blur` | Low distortion, high blur | Wiener + spread-guided refinement |
| `distortion` | High distortion, low blur | Wiener → geometric unwarp |
| `both` | Significant both | Full correction |
| `auto` | Let DSO decide | Based on measured WDC values |

## Results

Tested on real graphene TEM data:
```
Mean displacement: 0.56 px (low distortion)
Mean spread: 3.73 px (blur dominant)
Mode selected: blur

HF Energy:  +914.7%
Contrast:   +219.0%
```

## Supported Materials

- **Graphene** - Hexagonal lattice, a = 2.46 Å

Extensible to any material with known lattice geometry.

## Citation
```
Garrett, J. (2025). DSO Framework for Geometric Image Recovery.
https://github.com/garrjo/dso-imaging
```

## License

MIT
