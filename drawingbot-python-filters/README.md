# DrawingBot V3 Image Filters - Python Implementation

Complete Python implementation of all **63 image filters** from the DrawingBot V3 manual, organized into 10 categories.

## Overview

Since the DrawingBot V3 manual only provides filter names without specific parameters, this implementation uses **standard image processing parameters** based on OpenCV and PIL conventions. Each filter has been carefully designed with appropriate parameter ranges.

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy` - Array operations
- `Pillow` - Image loading/saving
- `opencv-python` - Advanced image processing
- `scipy` - Signal processing functions

## Usage

### Basic Example

```python
from image_filters import ImageFilterProcessor, GaussianBlurParams

# Initialize processor
processor = ImageFilterProcessor(use_gpu=False)

# Load image
from PIL import Image
image = Image.open("input.jpg").convert("RGB")

# Apply Gaussian Blur
params = GaussianBlurParams(radius=5.0)
result = processor.apply_filter(image, "Gaussian Blur", params)

# Save result
result.save("output.jpg")
```

### Applying Multiple Filters

```python
from image_filters import apply_filters_to_image, GaussianBlurParams, ThresholdParams

filters = [
    ("Gaussian Blur", GaussianBlurParams(radius=3.0)),
    ("Threshold", ThresholdParams(threshold=128)),
]

apply_filters_to_image(
    image_path="input.jpg",
    filters=filters,
    output_path="output.jpg"
)
```

### Creating Parameters Programmatically

```python
from image_filters import create_filter_params

# Create parameters with defaults
params = create_filter_params("Gaussian Blur")

# Create parameters with custom values
params = create_filter_params(
    "Gaussian Blur",
    radius=10.0
)
```

## Filter Categories

### 1. Borders (2 filters)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Dirty Border | Adds random noise/artifacts around edges | `border_width`, `noise_intensity` |
| Custom Overlay | Applies B&W overlay image | `overlay_path`, `opacity`, `blend_mode` |

### 2. Blur (14 filters)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Box Blur | Simple averaging blur | `radius` |
| Emboss Edges | Creates embossed edge effect | `angle`, `elevation`, `strength` |
| Gaussian Blur | Smooth Gaussian kernel blur | `radius`, `sigma` |
| Glow | Soft glow effect | `radius`, `intensity`, `color` |
| High Pass | Preserves high frequency details | `radius`, `strength` |
| Lens Blur | Simulates camera lens blur | `radius`, `blade_count`, `rotation` |
| Maximum | Expands bright areas (dilation) | `radius` |
| Median | Noise reduction preserving edges | `radius` |
| Minimum | Expands dark areas (erosion) | `radius` |
| Motion Blur Fast | Quick directional blur | `angle`, `distance` |
| Motion Blur Slow | Smooth directional blur | `angle`, `distance`, `samples` |
| Sharpen | Enhances edge contrast | `strength`, `radius` |
| Simple Blur | Basic uniform blur | `radius` |
| Smart Blur | Blurs while preserving edges | `radius`, `threshold`, `quality` |
| Unsharp Mask | Sharpens by subtracting blur | `radius`, `amount`, `threshold` |

### 3. Colors (13 filters)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Adjust HSB | Modifies Hue, Saturation, Brightness | `hue_shift`, `saturation_scale`, `brightness_scale` |
| Adjust RGB | Modifies RGB channels | `red_scale`, `green_scale`, `blue_scale` |
| Contrast | Adjusts image contrast | `factor` |
| Exposure | Adjusts exposure level | `exposure` |
| Gain | Multiplies pixel values | `gain` |
| Gamma | Gamma correction | `gamma` |
| Gray Out | Converts to grayscale | `opacity`, `method` |
| Invert | Inverts colors | `channel` |
| Levels | Adjusts input/output levels | `input_black`, `input_white`, `gamma` |
| Mix Channels | Remaps color channels | `red_source`, `green_source`, `blue_source` |
| Posterize | Reduces color levels | `levels` |
| Quantize | Reduces color palette | `colors`, `method`, `dither` |
| Rescale | Rescales brightness range | `scale_min`, `scale_max` |
| Solarize | Inverts colors above threshold | `threshold` |
| Transparency | Adjusts alpha channel | `alpha`, `fill_color` |

### 4. Distort (7 filters)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Diffuse | Randomly displaces pixels | `amount`, `iterations` |
| Displace | Displaces using displacement map | `map_path`, `scale_x`, `scale_y` |
| Kaleidoscope | Creates kaleidoscopic pattern | `segments`, `center_x`, `center_y` |
| Marble | Creates marble-like distortion | `vein_size`, `vein_strength`, `turbulence` |
| Ripple | Creates wave distortion | `amplitude`, `wavelength`, `direction` |
| Shear | Skews image along axis | `x_shear`, `y_shear` |
| Swim | Creates water-like distortion | `amount`, `speed`, `time` |

### 5. Edges (2 filters)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Detect Edges | Edge detection | `method`, `threshold1`, `threshold2` |
| Laplace | Laplacian edge detection | `kernel_size`, `scale`, `delta` |

### 6. Effects (4 filters)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Chrome | Creates chrome/metallic effect | `detail`, `smoothness` |
| Feedback | Creates feedback/echo effect | `offset_x`, `offset_y`, `decay` |
| Glint | Adds lens flare/glare | `intensity`, `size`, `rays` |
| Mirror | Mirrors image | `axis`, `center_x`, `center_y` |

### 7. Keying (1 filter)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Chroma Key | Removes background by color | `key_color`, `tolerance`, `softness` |

### 8. Pixellate (3 filters)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Color Halftone | CMYK halftone dots | `dot_size`, `angle_cyan`, `angle_magenta` |
| Crystallize | Creates crystal-like facets | `cell_size`, `randomness` |
| Pointillize | Creates pointillist effect | `dot_size`, `coverage`, `randomness` |

### 9. Render (1 filter)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Scratches | Adds film scratch effect | `count`, `length`, `thickness`, `opacity` |

### 10. Stylize (13 filters)
| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| Noise | Adds random noise | `amount`, `distribution`, `monochrome` |
| Contours | Draws contour lines | `levels`, `line_width`, `fill` |
| Dissolve | Random pixel dissolution | `amount` |
| Drop Shadow | Adds drop shadow | `offset_x`, `offset_y`, `blur_radius` |
| Emboss | Creates embossed relief | `angle`, `elevation`, `depth` |
| Flare | Adds lens flare | `intensity`, `size`, `position_x` |
| Oil | Simulates oil painting | `brush_size`, `levels`, `neighborhood` |
| Rays | Creates light rays | `center_x`, `ray_count`, `length` |
| Shape Burst | Bursts from shapes | `shape`, `burst_size`, `density` |
| Sparkle | Adds sparkle effects | `count`, `size`, `brightness` |
| Stamp | Creates rubber stamp effect | `texture_path`, `scale`, `opacity` |
| Threshold | Binary thresholding | `threshold`, `max_value`, `method` |

## GPU Acceleration

Enable GPU acceleration using CuPy:

```python
processor = ImageFilterProcessor(use_gpu=True)
```

Note: Requires `cupy` installation (`pip install cupy-cuda11x` or appropriate version for your CUDA).

## Integration with DrawingBot PFM

This image filter module complements the Path Finding Module (PFM) implementation:

```python
from drawingbot import DrawingBotV3
from image_filters import ImageFilterProcessor, GaussianBlurParams

# Pre-process image with filters
filter_processor = ImageFilterProcessor()
image = Image.open("input.jpg")
image = filter_processor.apply_filter(image, "Gaussian Blur", GaussianBlurParams(radius=3.0))
image.save("preprocessed.jpg")

# Then use with DrawingBot PFM
bot = DrawingBotV3(width=800, height=600)
bot.load_image("preprocessed.jpg")
paths = bot.generate("Sketch Lines", sketch_params)
```

## Parameter Ranges

All parameters include recommended ranges in their docstrings. For example:
- `radius: float = 2.0  # 0.1-50.0`
- `threshold: int = 128  # 0-255`
- `angle: float = 0.0  # 0-360`

These ranges are based on standard image processing practices and provide sensible defaults.

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.
