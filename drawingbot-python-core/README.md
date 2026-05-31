# DrawingBot V3 - Post-Processing Systems

Complete Python implementation of the four final systems in the DrawingBot V3 pipeline:

1. **Masking System** (Spatial Filtering)
2. **Pen Settings & Color Separation**
3. **Path Optimization**
4. **Hardware Export** (GCode, HPGL, SVG)

All parameters match the DrawingBot V3 specification exactly.

## Features

### 1. Masking System
- **Mask Types**: ADD (draw inside), SUBTRACT (erase inside)
- **Shapes**: Rectangle, Circle, Star, X, SVG Path
- **Transformations**: Rotation (0-360°), Skew X/Y, Translation
- **Soft Clip**: Organic avoidance vs mathematical slicing

```python
from post_processing import MaskingSystem, Mask, MaskType, MaskShape

masking = MaskingSystem(
    enable_masking=True,
    soft_clip=False,
    masks=[
        Mask(
            mask_type=MaskType.ADD,
            shape=MaskShape.RECTANGLE,
            start_x=0, start_y=0,
            width=50, height=50,
            rotation=45.0
        ),
        Mask(
            mask_type=MaskType.SUBTRACT,
            shape=MaskShape.CIRCLE,
            start_x=25, start_y=25,
            width=20, height=20
        )
    ]
)

masked_paths = masking.apply_to_paths(raw_paths)
```

### 2. Pen Settings & Color Separation
- **Distribution Types**: EvenWeighted, RandomWeighted, LuminanceWeighted, SinglePen
- **Distribution Order**: DarkestFirst, LightestFirst, Displayed, Reversed
- **Color Separation**: Default, CMYK, ColourMatch
- **Delta-E Color Matching**: Configurable accuracy (95-100)

```python
from post_processing import PenSettings, Pen, DistributionType, DistributionOrder

pen_settings = PenSettings(
    distribution_type=DistributionType.EVEN_WEIGHTED,
    distribution_order=DistributionOrder.DARKEST_FIRST,
    pens=[
        Pen(name="Black FineLiner", color=(0, 0, 0, 255), weight=1.0, pen_width=0.5),
        Pen(name="Red FineLiner", color=(255, 0, 0, 255), weight=0.8, pen_width=0.3),
        Pen(name="Blue FineLiner", color=(0, 0, 255, 255), weight=0.6, pen_width=0.3),
    ]
)

assignments = pen_settings.assign_paths_to_pens(paths, image_luminance)
```

### 3. Path Optimization
- **Douglas-Peucker Simplification**: Reduce vertex count within tolerance
- **Path Merging**: Connect endpoints within distance threshold
- **Path Filtering**: Remove paths shorter than minimum length
- **Nearest-Neighbor Sorting**: Minimize pen-up travel time using STRTree
- **Multipass**: Repeat geometry N times for poor ink flow

```python
from post_processing import PathOptimization

optimizer = PathOptimization(
    enable_simplifying=True,
    simplify_tolerance=0.5,
    enable_merging=True,
    merge_distance=2.0,
    enable_filtering=True,
    min_path_length=5.0,
    enable_sorting=True,
    multipass=1
)

optimized_paths = optimizer.optimize(raw_paths)
```

### 4. Hardware Export
#### G-Code Export
- Configurable offsets and center zero point
- Custom start/end routines
- Pen up/down commands (M3 S0/S90)
- Layer comments with wildcard support
- Comment styles: Brackets (), Semicolons ;, None

```python
from post_processing import HardwareExport, GCodeConfig, CommentType

exporter = HardwareExport(
    gcode_config=GCodeConfig(
        offset_x=10.0,
        offset_y=10.0,
        center_zero_point=False,
        comment_type=CommentType.SEMICOLONS,
        gcode_pen_down="M3 S90",
        gcode_pen_up="M3 S0",
        feed_rate=1000.0
    )
)

gcode = exporter.export_gcode(optimized_paths, "output.gcode", pen_assignments)
```

#### HPGL Export
- Hard clip boundaries (40 units = 1mm)
- Rotation (0°, 90°, 180°, 270°, AUTO)
- Axis mirroring
- Pen velocity and force control
- Initial pen selection

```python
from post_processing import HPGLConfig, HPGLRotation

exporter.hpgl_config = HPGLConfig(
    hard_clip_min_x=0,
    hard_clip_max_x=10000,
    rotation=HPGLRotation.ROT_0,
    pen_velocity=30,
    pen_force=30,
    initial_pen=1
)

hpgl = exporter.export_hpgl(paths, "output.hpgl")
```

#### SVG Export
- Configurable viewBox and units
- Metadata inclusion
- Layer grouping by pen
- Stroke width control

```python
from post_processing import SVGConfig

exporter.svg_config = SVGConfig(
    viewBox_width=800.0,
    viewBox_height=600.0,
    stroke_width=0.5,
    units="mm"
)

svg = exporter.export_svg(paths, "output.svg", pen_assignments)
```

## Complete Pipeline Example

```python
from post_processing import (
    MaskingSystem, Mask, MaskType, MaskShape,
    PenSettings, Pen, DistributionType, DistributionOrder,
    PathOptimization,
    HardwareExport, GCodeConfig
)

# Sample paths from PFM algorithms
sample_paths = [
    [(0, 0), (10, 10), (20, 0), (30, 10)],
    [(5, 5), (15, 15), (25, 5)],
    [(0, 20), (30, 20), (30, 40), (0, 40), (0, 20)],
]

# 1. Apply masking
masking = MaskingSystem(
    enable_masking=True,
    masks=[Mask(mask_type=MaskType.ADD, width=50, height=50)]
)
masked_paths = masking.apply_to_paths(sample_paths)

# 2. Assign to pens
pen_settings = PenSettings(
    distribution_type=DistributionType.EVEN_WEIGHTED,
    pens=[
        Pen(name="Black", color=(0, 0, 0, 255)),
        Pen(name="Red", color=(255, 0, 0, 255)),
    ]
)
pen_assignments = pen_settings.assign_paths_to_pens(masked_paths)

# 3. Optimize paths
optimizer = PathOptimization(
    enable_simplifying=True,
    simplify_tolerance=0.3,
    enable_sorting=True
)
optimized_paths = optimizer.optimize(masked_paths)

# 4. Export to G-code
exporter = HardwareExport(
    gcode_config=GCodeConfig(
        offset_x=10.0,
        comment_type=CommentType.SEMICOLONS
    )
)
gcode = exporter.export_gcode(optimized_paths, "output.gcode", pen_assignments)
svg = exporter.export_svg(optimized_paths, "output.svg", pen_assignments)

print(f"Generated {len(gcode)} bytes G-code")
print(f"Generated {len(svg)} bytes SVG")
```

## Running the Example

```bash
cd /workspace/drawingbot-python-core
python post_processing.py
```

Output:
```
After masking: 3 paths
Pen assignments: [('Black FineLiner', 1), ('Red FineLiner', 2)]
After optimization: 3 paths
Generated G-code: 559 bytes
Generated SVG: 639 bytes

✅ Complete pipeline executed successfully!
```

## Architecture Notes

### Data Structures
- All configurations use Python `@dataclass` for type safety
- Enums for all categorical parameters
- Lists of tuples `(x, y)` for path representation
- Dictionary mapping for pen assignments

### Algorithms Implemented
- **Point-in-polygon tests** for all mask shapes
- **Inverse transformation** for rotated/skewed masks
- **Douglas-Peucker** for path simplification
- **Nearest-neighbor greedy** for path sorting
- **Endpoint merging** with configurable tolerance

### C++ Translation Guide
For C++ implementation:
- Replace `dataclass` with `struct`
- Replace `Enum` with `enum class`
- Replace `List[Tuple[float, float]]` with `std::vector<std::pair<double, double>>`
- Replace `Dict[str, List[...]]` with `std::unordered_map<std::string, std::vector<...>>`
- Use `Eigen` or similar for transformation matrices
- Use `Rtree` library for STRTree spatial indexing

## Files
- `post_processing.py` - Complete implementation (1033 lines)
- `output.gcode` - Example G-code output
- `output.svg` - Example SVG output

## Dependencies
- Python 3.8+
- No external dependencies (uses only standard library)
