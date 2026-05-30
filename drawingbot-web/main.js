import * as THREE from 'three';

// Main application state
const state = {
    isDrawing: false,
    brushSize: 2.0,
    color: new THREE.Color('#e94560'),
    opacity: 1.0,
    flow: 0.5,
    strokeType: 'continuous',
    density: 50,
    maxPoints: 50000,
    points: [],
    strokes: [],
    currentStroke: []
};

// Three.js setup
let scene, camera, renderer, canvas;
let lineGeometry, lineMaterial, lineMesh;
let pointGeometry, pointMaterial, pointsMesh;
let clock, fpsCounter;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

// Initialize the application
function init() {
    const container = document.getElementById('canvas-container');
    
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    
    // Camera - Orthographic for 2D drawing
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.OrthographicCamera(
        -container.clientWidth / 2,
        container.clientWidth / 2,
        container.clientHeight / 2,
        -container.clientHeight / 2,
        0.1,
        1000
    );
    camera.position.z = 10;
    
    // Renderer with GPU optimization
    renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        powerPreference: 'high-performance'
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    canvas = renderer.domElement;
    
    // Create line geometry using BufferGeometry for GPU efficiency
    lineGeometry = new THREE.BufferGeometry();
    const maxLineVertices = state.maxPoints * 2;
    const linePositions = new Float32Array(maxLineVertices * 3);
    const lineColors = new Float32Array(maxLineVertices * 3);
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
    lineGeometry.setDrawRange(0, 0);
    
    // Line material with vertex colors
    lineMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        linewidth: 2,
        transparent: true,
        opacity: state.opacity
    });
    
    lineMesh = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lineMesh);
    
    // Point geometry for stippling
    pointGeometry = new THREE.BufferGeometry();
    const maxPointVertices = state.maxPoints;
    const pointPositions = new Float32Array(maxPointVertices * 3);
    const pointColors = new Float32Array(maxPointVertices * 3);
    const pointSizes = new Float32Array(maxPointVertices);
    pointGeometry.setAttribute('position', new THREE.BufferAttribute(pointPositions, 3));
    pointGeometry.setAttribute('color', new THREE.BufferAttribute(pointColors, 3));
    pointGeometry.setAttribute('size', new THREE.BufferAttribute(pointSizes, 1));
    pointGeometry.setDrawRange(0, 0);
    
    // Custom shader material for points with variable size
    pointMaterial = new THREE.ShaderMaterial({
        vertexShader: `
            attribute float size;
            varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            void main() {
                float r = distance(gl_PointCoord, vec2(0.5, 0.5));
                if (r > 0.5) discard;
                gl_FragColor = vec4(vColor, 1.0);
            }
        `,
        vertexColors: true,
        transparent: true,
        opacity: state.opacity
    });
    
    pointsMesh = new THREE.Points(pointGeometry, pointMaterial);
    scene.add(pointsMesh);
    
    // Clock for FPS tracking
    clock = new THREE.Clock();
    
    // Event listeners
    setupEventListeners(container);
    setupUIControls();
    
    // Start render loop
    animate();
}

// Setup mouse/touch event listeners
function setupEventListeners(container) {
    const getMousePos = (event) => {
        const rect = canvas.getBoundingClientRect();
        const clientX = event.touches ? event.touches[0].clientX : event.clientX;
        const clientY = event.touches ? event.touches[0].clientY : event.clientY;
        return {
            x: clientX - rect.left - container.clientWidth / 2,
            y: -(clientY - rect.top - container.clientHeight / 2)
        };
    };
    
    const startDrawing = (event) => {
        event.preventDefault();
        state.isDrawing = true;
        state.currentStroke = [];
        const pos = getMousePos(event);
        state.currentStroke.push({ x: pos.x, y: pos.y });
    };
    
    const draw = (event) => {
        if (!state.isDrawing) return;
        event.preventDefault();
        
        const pos = getMousePos(event);
        state.currentStroke.push({ x: pos.x, y: pos.y });
        
        // Add points based on stroke type
        addPointsToStroke(pos);
    };
    
    const stopDrawing = () => {
        if (state.isDrawing) {
            state.isDrawing = false;
            if (state.currentStroke.length > 0) {
                state.strokes.push([...state.currentStroke]);
            }
            state.currentStroke = [];
        }
    };
    
    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
    
    // Touch events
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);
    
    // Window resize
    window.addEventListener('resize', () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        camera.left = -width / 2;
        camera.right = width / 2;
        camera.top = height / 2;
        camera.bottom = -height / 2;
        camera.updateProjectionMatrix();
        
        renderer.setSize(width, height);
    });
}

// Add points based on stroke type and settings
function addPointsToStroke(pos) {
    const strokeType = state.strokeType;
    const density = state.density / 100;
    
    if (strokeType === 'continuous') {
        state.points.push({
            x: pos.x,
            y: pos.y,
            color: state.color.clone(),
            size: state.brushSize
        });
    } else if (strokeType === 'stippling') {
        // Add dots based on density
        if (Math.random() < density) {
            state.points.push({
                x: pos.x,
                y: pos.y,
                color: state.color.clone(),
                size: state.brushSize * 2
            });
        }
    } else if (strokeType === 'hatching' || strokeType === 'crosshatch' || strokeType === 'zigzag') {
        // Generate pattern based on movement
        const lastPoint = state.points[state.points.length - 1];
        if (!lastPoint || Math.abs(pos.x - lastPoint.x) > state.brushSize * density || 
            Math.abs(pos.y - lastPoint.y) > state.brushSize * density) {
            state.points.push({
                x: pos.x,
                y: pos.y,
                color: state.color.clone(),
                size: state.brushSize
            });
        }
    }
    
    // Limit points
    if (state.points.length > state.maxPoints) {
        state.points.shift();
    }
    
    updateGeometry();
    updateStats();
}

// Update GPU buffers with new geometry
function updateGeometry() {
    const linePositions = lineGeometry.attributes.position.array;
    const lineColors = lineGeometry.attributes.color.array;
    const pointPositions = pointGeometry.attributes.position.array;
    const pointColors = pointGeometry.attributes.color.array;
    const pointSizes = pointGeometry.attributes.size.array;
    
    // Update lines (continuous mode only)
    let lineIndex = 0;
    let pointIndex = 0;
    
    for (let i = 0; i < state.points.length - 1; i++) {
        if (state.strokeType === 'continuous') {
            const p1 = state.points[i];
            const p2 = state.points[i + 1];
            
            linePositions[lineIndex * 3] = p1.x;
            linePositions[lineIndex * 3 + 1] = p1.y;
            linePositions[lineIndex * 3 + 2] = 0;
            
            linePositions[lineIndex * 3 + 3] = p2.x;
            linePositions[lineIndex * 3 + 4] = p2.y;
            linePositions[lineIndex * 3 + 5] = 0;
            
            lineColors[lineIndex * 3] = p1.color.r;
            lineColors[lineIndex * 3 + 1] = p1.color.g;
            lineColors[lineIndex * 3 + 2] = p1.color.b * state.opacity;
            
            lineColors[lineIndex * 3 + 3] = p2.color.r;
            lineColors[lineIndex * 3 + 4] = p2.color.g;
            lineColors[lineIndex * 3 + 5] = p2.color.b * state.opacity;
            
            lineIndex++;
        }
    }
    
    // Update points
    for (let i = 0; i < state.points.length; i++) {
        const p = state.points[i];
        pointPositions[pointIndex * 3] = p.x;
        pointPositions[pointIndex * 3 + 1] = p.y;
        pointPositions[pointIndex * 3 + 2] = 0;
        
        pointColors[pointIndex * 3] = p.color.r;
        pointColors[pointIndex * 3 + 1] = p.color.g;
        pointColors[pointIndex * 3 + 2] = p.color.b * state.opacity;
        
        pointSizes[pointIndex] = p.size;
        pointIndex++;
    }
    
    lineGeometry.attributes.position.needsUpdate = true;
    lineGeometry.attributes.color.needsUpdate = true;
    lineGeometry.setDrawRange(0, lineIndex * 2);
    
    pointGeometry.attributes.position.needsUpdate = true;
    pointGeometry.attributes.color.needsUpdate = true;
    pointGeometry.attributes.size.needsUpdate = true;
    pointGeometry.setDrawRange(0, pointIndex);
}

// Setup UI controls
function setupUIControls() {
    // Brush size
    const brushSizeSlider = document.getElementById('brush-size');
    const brushSizeValue = document.getElementById('brush-size-value');
    brushSizeSlider.addEventListener('input', (e) => {
        state.brushSize = parseFloat(e.target.value);
        brushSizeValue.textContent = state.brushSize.toFixed(1);
    });
    
    // Color
    const colorPicker = document.getElementById('brush-color');
    colorPicker.addEventListener('input', (e) => {
        state.color = new THREE.Color(e.target.value);
    });
    
    // Opacity
    const opacitySlider = document.getElementById('opacity');
    const opacityValue = document.getElementById('opacity-value');
    opacitySlider.addEventListener('input', (e) => {
        state.opacity = parseFloat(e.target.value);
        opacityValue.textContent = state.opacity.toFixed(2);
        lineMaterial.opacity = state.opacity;
        pointMaterial.opacity = state.opacity;
    });
    
    // Flow
    const flowSlider = document.getElementById('flow');
    const flowValue = document.getElementById('flow-value');
    flowSlider.addEventListener('input', (e) => {
        state.flow = parseFloat(e.target.value);
        flowValue.textContent = state.flow.toFixed(2);
    });
    
    // Stroke type
    const strokeTypeSelect = document.getElementById('stroke-type');
    strokeTypeSelect.addEventListener('change', (e) => {
        state.strokeType = e.target.value;
    });
    
    // Density
    const densitySlider = document.getElementById('density');
    const densityValue = document.getElementById('density-value');
    densitySlider.addEventListener('input', (e) => {
        state.density = parseInt(e.target.value);
        densityValue.textContent = state.density;
    });
    
    // Clear canvas
    document.getElementById('clear-canvas').addEventListener('click', clearCanvas);
    
    // Export PNG
    document.getElementById('export-png').addEventListener('click', exportPNG);
    
    // Export SVG
    document.getElementById('export-svg').addEventListener('click', exportSVG);
    
    // Max points
    const maxPointsSlider = document.getElementById('max-points');
    const maxPointsValue = document.getElementById('max-points-value');
    maxPointsSlider.addEventListener('input', (e) => {
        state.maxPoints = parseInt(e.target.value);
        maxPointsValue.textContent = state.maxPoints;
    });
    
    // Optimize paths
    document.getElementById('optimize').addEventListener('click', optimizePaths);
}

// Clear the canvas
function clearCanvas() {
    state.points = [];
    state.strokes = [];
    state.currentStroke = [];
    updateGeometry();
    updateStats();
}

// Export as PNG
function exportPNG() {
    renderer.render(scene, camera);
    const dataURL = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = 'drawingbot-export.png';
    link.href = dataURL;
    link.click();
}

// Export as SVG
function exportSVG() {
    let svgContent = `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">`;
    
    if (state.strokeType === 'continuous') {
        let pathData = '';
        state.points.forEach((point, index) => {
            const x = point.x + 400; // Center offset
            const y = 300 - point.y; // Flip Y
            if (index === 0) {
                pathData += `M ${x} ${y}`;
            } else {
                pathData += ` L ${x} ${y}`;
            }
        });
        
        if (pathData) {
            const colorStr = state.color.getStyle();
            svgContent += `<path d="${pathData}" stroke="${colorStr}" stroke-width="${state.brushSize}" fill="none" opacity="${state.opacity}"/>`;
        }
    } else if (state.strokeType === 'stippling') {
        const colorStr = state.color.getStyle();
        state.points.forEach(point => {
            const x = point.x + 400;
            const y = 300 - point.y;
            svgContent += `<circle cx="${x}" cy="${y}" r="${point.size / 2}" fill="${colorStr}" opacity="${state.opacity}"/>`;
        });
    }
    
    svgContent += `</svg>`;
    
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = 'drawingbot-export.svg';
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
}

// Optimize paths (simplified Traveling Salesman approximation)
function optimizePaths() {
    if (state.points.length < 2) return;
    
    // Simple nearest neighbor optimization
    const optimized = [state.points[0]];
    const remaining = state.points.slice(1);
    
    while (remaining.length > 0) {
        const lastPoint = optimized[optimized.length - 1];
        let nearestIndex = 0;
        let nearestDist = Infinity;
        
        for (let i = 0; i < remaining.length; i++) {
            const dist = Math.sqrt(
                Math.pow(remaining[i].x - lastPoint.x, 2) +
                Math.pow(remaining[i].y - lastPoint.y, 2)
            );
            if (dist < nearestDist) {
                nearestDist = dist;
                nearestIndex = i;
            }
        }
        
        optimized.push(remaining[nearestIndex]);
        remaining.splice(nearestIndex, 1);
    }
    
    state.points = optimized;
    updateGeometry();
}

// Update statistics display
function updateStats() {
    document.getElementById('point-count').textContent = state.points.length;
    
    // Estimate GPU memory (rough approximation)
    const gpuMemory = (state.points.length * 6 * 4) / (1024 * 1024); // positions + colors
    document.getElementById('gpu-memory').textContent = gpuMemory.toFixed(2);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Calculate FPS
    const currentTime = clock.getElapsedTime();
    frameCount++;
    
    if (currentTime - lastFrameTime >= 1.0) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = currentTime;
        document.getElementById('fps').textContent = fps;
    }
    
    renderer.render(scene, camera);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
