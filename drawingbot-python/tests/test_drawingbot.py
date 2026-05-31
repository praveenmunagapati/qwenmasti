"""
Test suite for DrawingBot V3 Python implementation
Tests all PFM parameter classes and core functionality
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drawingbot import (
    DrawingBotV3,
    # Sketch PFMs
    SketchLinesParams,
    SketchCurvesParams,
    SketchSquaresParams,
    SketchQuadBeziersParams,
    SketchCubicBeziersParams,
    SketchCatmullRomsParams,
    SketchShapesParams,
    SketchSobelEdgesParams,
    SketchWavesParams,
    SketchFlowFieldParams,
    SketchSuperformulaParams,
    SketchSweepingCurvesParams,
    # Streamline PFMs
    StreamlinesEdgeFieldParams,
    StreamlinesFlowFieldParams,
    StreamlinesSuperformulaParams,
    # Spiral PFMs
    SpiralSawtoothParams,
    SpiralCircularScribblesParams,
    SpiralType,
    # Hatch PFMs
    HatchSawtoothParams,
    HatchCircularScribblesParams,
    # Adaptive PFMs
    AdaptiveBaseParams,
    AdaptiveCircularScribblesParams,
    AdaptiveShapesParams,
    AdaptiveTriangulationParams,
    AdaptiveTreeParams,
    AdaptiveStipplingParams,
    AdaptiveDashesParams,
    AdaptiveLettersParams,
    AdaptiveDiagramParams,
    AdaptiveTSPParams,
    ShapeType,
    VoronoiStyle,
    OrderType,
    FontStyle,
)


class TestParameterClasses:
    """Test all PFM parameter dataclasses"""
    
    def test_sketch_lines_params(self):
        """Test SketchLinesParams with default values"""
        params = SketchLinesParams()
        assert params.plottingResolution == 0.5
        assert params.lineDensity == 50.0
        assert params.lineMinLength == 10
        assert params.lineMaxLength == 100
    
    def test_sketch_curves_params(self):
        """Test SketchCurvesParams inherits from SketchLinesParams"""
        params = SketchCurvesParams()
        assert params.curveTension == 0.5
        assert params.lineDensity == 50.0  # Inherited
    
    def test_sketch_waves_params(self):
        """Test SketchWavesParams with wave types"""
        params = SketchWavesParams(
            waveOffsetX=50.0,
            waveOffsetY=30.0,
            waveDivisorX=200.0
        )
        assert params.waveTypeX.value == "Sin"
        assert params.waveOffsetX == 50.0
    
    def test_sketch_flow_field_params(self):
        """Test SketchFlowFieldParams"""
        params = SketchFlowFieldParams(
            xFrequency=0.5,
            yFrequency=0.3,
            amplitude=0.8
        )
        assert params.xFrequency == 0.5
        assert params.amplitude == 0.8
    
    def test_sketch_superformula_params(self):
        """Test SketchSuperformulaParams"""
        params = SketchSuperformulaParams(
            centreX=75.0,
            frequency=5.0,
            curvature=30.0
        )
        assert params.centreX == 75.0
        assert params.frequency == 5.0
    
    def test_spiral_sawtooth_params(self):
        """Test SpiralSawtoothParams"""
        params = SpiralSawtoothParams(
            spiralType=SpiralType.ARCHIMEDEAN,
            spiralSize=60.0,
            ringSpacing=8.0
        )
        assert params.spiralType == SpiralType.ARCHIMEDEAN
        assert params.spiralSize == 60.0
    
    def test_spiral_circular_scribbles_params(self):
        """Test SpiralCircularScribblesParams"""
        params = SpiralCircularScribblesParams(
            minRadius=5.0,
            maxRadius=40.0,
            angularVelocity=90.0
        )
        assert params.minRadius == 5.0
        assert params.angularVelocity == 90.0
    
    def test_hatch_sawtooth_params(self):
        """Test HatchSawtoothParams"""
        params = HatchSawtoothParams(
            lineSpacing=15.0,
            angle=60.0,
            crosshatch=True
        )
        assert params.lineSpacing == 15.0
        assert params.crosshatch is True
    
    def test_adaptive_shapes_params(self):
        """Test AdaptiveShapesParams with shape types"""
        params = AdaptiveShapesParams(
            shapeType=ShapeType.STAR,
            minRotation=45.0,
            maxRotation=315.0
        )
        assert params.shapeType == ShapeType.STAR
        assert params.minRotation == 45.0
    
    def test_adaptive_letters_params(self):
        """Test AdaptiveLettersParams with order and font style"""
        params = AdaptiveLettersParams(
            order=OrderType.SEQUENCED,
            font="Helvetica",
            style=FontStyle.BOLD
        )
        assert params.order == OrderType.SEQUENCED
        assert params.style == FontStyle.BOLD
    
    def test_adaptive_diagram_params(self):
        """Test AdaptiveDiagramParams with voronoi styles"""
        params = AdaptiveDiagramParams(
            voronoiStyle=VoronoiStyle.SMOOTH
        )
        assert params.voronoiStyle == VoronoiStyle.SMOOTH


class TestDrawingBotCore:
    """Test core DrawingBot functionality"""
    
    def test_initialization_cpu(self):
        """Test DrawingBot initialization with CPU"""
        bot = DrawingBotV3(width=800, height=600, use_gpu=False)
        assert bot.width == 800
        assert bot.height == 600
        assert bot.use_gpu is False
        assert bot.xp == np
    
    def test_initialization_with_test_image(self):
        """Test DrawingBot with test image data"""
        bot = DrawingBotV3(width=100, height=100, use_gpu=False)
        test_image = np.random.rand(100, 100).astype(np.float32)
        bot.image_data = test_image
        assert bot.image_data.shape == (100, 100)
    
    def test_generate_sketch_lines(self):
        """Test sketch lines generation"""
        bot = DrawingBotV3(width=100, height=100, use_gpu=False)
        bot.image_data = np.random.rand(100, 100).astype(np.float32)
        
        params = SketchLinesParams(
            lineMinLength=5,
            lineMaxLength=50
        )
        
        paths = bot.generate_sketch_lines(params)
        assert isinstance(paths, list)
    
    def test_save_svg(self, tmp_path):
        """Test SVG export"""
        bot = DrawingBotV3(width=100, height=100, use_gpu=False)
        bot.image_data = np.ones((100, 100), dtype=np.float32) * 0.5
        bot.algorithm_name = "Sketch Lines"
        
        # Create some test paths
        bot.paths = [
            [(10, 10), (20, 20), (30, 30)],
            [(50, 50), (60, 60)]
        ]
        
        output_file = tmp_path / "test_output.svg"
        bot.save_svg(output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert '<svg' in content
        assert 'path' in content
    
    def test_save_gcode(self, tmp_path):
        """Test G-code export"""
        bot = DrawingBotV3(width=100, height=100, use_gpu=False)
        bot.algorithm_name = "Sketch Lines"
        
        # Create some test paths
        bot.paths = [
            [(10, 10), (20, 20), (30, 30)]
        ]
        
        output_file = tmp_path / "test_output.gcode"
        bot.save_gcode(output_file, feed_rate=1500.0, z_lift=3.0)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert 'G21' in content
        assert 'G90' in content
        assert 'F1500.0' in content


class TestUtilityFunctions:
    """Test utility mathematical functions"""
    
    def test_archimedean_spiral(self):
        """Test Archimedean spiral calculation"""
        from drawingbot import archimedean_spiral
        
        params = SpiralSawtoothParams(
            centreX=50.0,
            centreY=50.0,
            ringSpacing=10.0
        )
        
        x, y = archimedean_spiral(0.0, params)
        assert x == 50.0 + params.ringSpacing / (2 * np.pi) * 0
        assert y == 50.0
    
    def test_parabolic_spiral(self):
        """Test parabolic spiral calculation"""
        from drawingbot import parabolic_spiral
        
        params = SpiralSawtoothParams(
            centreX=50.0,
            centreY=50.0,
            ringSpacing=10.0
        )
        
        x, y = parabolic_spiral(np.pi, params)
        assert isinstance(x, float)
        assert isinstance(y, float)


class TestParameterRanges:
    """Test that parameters respect specified ranges"""
    
    def test_plotting_resolution_range(self):
        """Test plottingResolution range (0.1-1.0)"""
        params = SketchLinesParams(plottingResolution=0.1)
        assert params.plottingResolution >= 0.1
        
        params = SketchLinesParams(plottingResolution=1.0)
        assert params.plottingResolution <= 1.0
    
    def test_angle_ranges(self):
        """Test angle parameter ranges (-360 to 360)"""
        params = SketchSquaresParams(startAngle=-360.0)
        assert params.startAngle >= -360.0
        
        params = SketchSquaresParams(startAngle=360.0)
        assert params.startAngle <= 360.0
    
    def test_line_length_ranges(self):
        """Test line length ranges (2-500)"""
        params = SketchLinesParams(lineMinLength=2)
        assert params.lineMinLength >= 2
        
        params = SketchLinesParams(lineMaxLength=500)
        assert params.lineMaxLength <= 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
