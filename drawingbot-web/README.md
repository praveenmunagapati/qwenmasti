# DrawingBot V3 Web Clone - GPU Accelerated

A GPU-accelerated web-based drawing application inspired by DrawingBot V3, built with Three.js and WebGL.

## Features

### 🎨 Drawing Tools
- **Continuous Line**: Smooth freehand drawing
- **Stippling**: Create dot-based artwork
- **Hatching**: Parallel line patterns
- **Cross-Hatching**: Intersecting line patterns  
- **Zigzag**: Zigzag pattern drawing

### ⚙️ Brush Controls
- Adjustable brush size (0.1 - 20)
- Color picker with full RGB spectrum
- Opacity control (0.1 - 1.0)
- Flow control for stroke density
- Pattern density adjustment

### 🚀 GPU Optimization
- Uses Three.js BufferGeometry for efficient GPU rendering
- Custom GLSL shaders for point rendering
- Real-time FPS monitoring
- GPU memory usage tracking
- Supports up to 500,000 points

### 📤 Export Options
- **PNG Export**: High-quality raster export
- **SVG Export**: Vector format for CNC/plotter use
- **Path Optimization**: Nearest-neighbor algorithm for efficient plotting paths

## Tech Stack

- **Three.js**: WebGL-based 3D/2D rendering
- **Vite**: Fast build tool and dev server
- **BufferGeometry**: GPU-optimized geometry handling
- **Custom Shaders**: GLSL vertex and fragment shaders for points

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

```bash
cd drawingbot-web
npm install
```

### Development

```bash
npm run dev
```

This starts a local development server with hot module replacement.

### Production Build

```bash
npm run build
```

Built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Usage

1. **Select Brush Type**: Choose from the stroke type dropdown
2. **Adjust Settings**: Modify brush size, color, opacity, and flow
3. **Draw**: Click and drag on the canvas to draw
4. **Export**: Use the export buttons to save your artwork
5. **Optimize**: Click "Optimize Paths" to minimize travel distance for plotters

## Performance Tips

- Keep point count under 50,000 for best performance
- Use lower density settings for faster drawing
- Enable path optimization before exporting for CNC machines
- Monitor FPS counter in top-left corner

## License

ISC

## Acknowledgments

Inspired by [DrawingBot V3](https://jungletools.com/drawingbot.html) - a popular software for generating toolpaths for plotters and CNC machines.
