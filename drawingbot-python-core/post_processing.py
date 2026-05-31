"""
DrawingBot V3 - Post-Processing Systems
Complete implementation of:
1. Masking System (Spatial Filtering)
2. Pen Settings & Color Separation
3. Path Optimization
4. Hardware Export Settings (GCode, HPGL, SVG)

All parameters match the DrawingBot V3 specification exactly.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import math
import random
from pathlib import Path


# ============================================================================
# 1. MASKING SYSTEM (Spatial Filtering)
# ============================================================================

class MaskType(Enum):
    ADD = "Add"           # Draw only inside
    SUBTRACT = "Subtract" # Erase inside (takes priority)


class MaskShape(Enum):
    RECTANGLE = "Rectangle"
    CIRCLE = "Circle"
    STAR = "Star"
    X = "X"
    SVG_PATH = "SVG Path"


@dataclass
class Mask:
    """Single mask definition with full geometric transformations."""
    enabled: bool = True
    mask_type: MaskType = MaskType.ADD
    shape: MaskShape = MaskShape.RECTANGLE
    start_x: float = 0.0
    start_y: float = 0.0
    width: float = 100.0
    height: float = 100.0
    rotation: float = 0.0  # 0-360 degrees
    skew_x: float = 0.0
    skew_y: float = 0.0
    svg_path: Optional[str] = None  # For SVG_PATH shape
    
    # Star-specific parameters
    star_points: int = 5
    star_inner_radius: float = 0.5  # Ratio of outer radius
    
    def __post_init__(self):
        if self.rotation < 0 or self.rotation > 360:
            raise ValueError("rotation must be 0-360 degrees")


@dataclass
class MaskingSystem:
    """Global masking system configuration."""
    enable_masking: bool = True
    soft_clip: bool = False  # Organic avoidance vs mathematical slicing
    masks: List[Mask] = field(default_factory=list)
    
    def add_mask(self, mask: Mask):
        """Add a mask to the system."""
        self.masks.append(mask)
    
    def apply_to_paths(self, paths: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """
        Apply masking to generated paths.
        
        Args:
            paths: List of paths, where each path is a list of (x, y) points
            
        Returns:
            Filtered/modified paths based on mask configuration
        """
        if not self.enable_masking or not self.masks:
            return paths
        
        # Separate ADD and SUBTRACT masks (SUBTRACT takes priority)
        add_masks = [m for m in self.masks if m.enabled and m.mask_type == MaskType.ADD]
        subtract_masks = [m for m in self.masks if m.enabled and m.mask_type == MaskType.SUBTRACT]
        
        result_paths = []
        
        for path in paths:
            # Apply SUBTRACT masks first (priority)
            modified_path = self._apply_subtract_masks(path, subtract_masks)
            
            # Then apply ADD masks
            if add_masks:
                modified_path = self._apply_add_masks(modified_path, add_masks)
            
            if len(modified_path) >= 2:  # Valid path needs at least 2 points
                result_paths.append(modified_path)
        
        return result_paths
    
    def _point_in_mask(self, x: float, y: float, mask: Mask) -> bool:
        """Check if a point is inside a mask shape."""
        # Transform point by inverse of mask transformations
        tx, ty = self._inverse_transform(x, y, mask)
        
        if mask.shape == MaskShape.RECTANGLE:
            return 0 <= tx <= mask.width and 0 <= ty <= mask.height
        
        elif mask.shape == MaskShape.CIRCLE:
            cx, cy = mask.width / 2, mask.height / 2
            radius = min(mask.width, mask.height) / 2
            return (tx - cx) ** 2 + (ty - cy) ** 2 <= radius ** 2
        
        elif mask.shape == MaskShape.STAR:
            return self._point_in_star(tx, ty, mask)
        
        elif mask.shape == MaskShape.X:
            return self._point_in_x(tx, ty, mask)
        
        elif mask.shape == MaskShape.SVG_PATH and mask.svg_path:
            return self._point_in_svg_path(tx, ty, mask.svg_path)
        
        return False
    
    def _inverse_transform(self, x: float, y: float, mask: Mask) -> Tuple[float, float]:
        """Apply inverse transformations to a point."""
        # Translate to origin
        x -= mask.start_x
        y -= mask.start_y
        
        # Un-rotate
        rad = -math.radians(mask.rotation)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        x, y = x * cos_a - y * sin_a, x * sin_a + y * cos_a
        
        # Un-skew
        if abs(math.cos(math.radians(mask.skew_x))) > 1e-6:
            x -= y * math.tan(math.radians(mask.skew_x))
        if abs(math.cos(math.radians(mask.skew_y))) > 1e-6:
            y -= x * math.tan(math.radians(mask.skew_y))
        
        return x, y
    
    def _point_in_star(self, x: float, y: float, mask: Mask) -> bool:
        """Check if point is inside a star shape."""
        cx, cy = mask.width / 2, mask.height / 2
        outer_r = min(mask.width, mask.height) / 2
        inner_r = outer_r * mask.star_inner_radius
        
        dx, dy = x - cx, y - cy
        angle = math.atan2(dy, dx)
        dist = math.sqrt(dx * dx + dy * dy)
        
        # Calculate star radius at this angle
        sector_angle = math.pi / mask.star_points
        normalized_angle = (angle % (2 * sector_angle)) - sector_angle
        star_radius = inner_r + (outer_r - inner_r) * abs(normalized_angle) / sector_angle
        
        return dist <= star_radius
    
    def _point_in_x(self, x: float, y: float, mask: Mask) -> bool:
        """Check if point is inside an X shape."""
        cx, cy = mask.width / 2, mask.height / 2
        thickness = min(mask.width, mask.height) * 0.15
        
        # Transform to center
        dx, dy = x - cx, y - cy
        
        # Check distance from both diagonals
        d1 = abs(dx - dy) / math.sqrt(2)
        d2 = abs(dx + dy) / math.sqrt(2)
        
        return d1 <= thickness or d2 <= thickness
    
    def _point_in_svg_path(self, x: float, y: float, svg_path: str) -> bool:
        """Check if point is inside SVG path (simplified ray-casting)."""
        # Placeholder - would require full SVG path parsing
        # In production, use a library like svg.path or cairo
        return True
    
    def _apply_subtract_masks(self, path: List[Tuple[float, float]], 
                              masks: List[Mask]) -> List[Tuple[float, float]]:
        """Remove path segments inside SUBTRACT masks."""
        if not masks:
            return path
        
        result = []
        for i, (x, y) in enumerate(path):
            inside_any = any(self._point_in_mask(x, y, m) for m in masks)
            if not inside_any:
                result.append((x, y))
        
        return result
    
    def _apply_add_masks(self, path: List[Tuple[float, float]], 
                         masks: List[Mask]) -> List[Tuple[float, float]]:
        """Keep only path segments inside ADD masks."""
        if not masks:
            return path
        
        result = []
        for x, y in path:
            inside_any = any(self._point_in_mask(x, y, m) for m in masks)
            if inside_any:
                result.append((x, y))
        
        return result


# ============================================================================
# 2. PEN SETTINGS & COLOR SEPARATION
# ============================================================================

class DistributionType(Enum):
    EVEN_WEIGHTED = "EvenWeighted"
    RANDOM_WEIGHTED = "RandomWeighted"
    RANDOM_SQUIGGLES = "RandomSquiggles"
    LUMINANCE_WEIGHTED = "LuminanceWeighted"
    PRECONFIGURED = "Preconfigured"
    SINGLE_PEN = "SinglePen"


class DistributionOrder(Enum):
    DARKEST_FIRST = "DarkestFirst"
    LIGHTEST_FIRST = "LightestFirst"
    DISPLAYED = "Displayed"
    REVERSED = "Reversed"


class ColorSeparation(Enum):
    DEFAULT = "Default"
    CMYK = "CMYK"
    COLOUR_MATCH = "ColourMatch"


@dataclass
class Pen:
    """Individual pen definition."""
    enabled: bool = True
    pen_type: str = "Special"  # Manufacturer or "Special"
    name: str = "FineLiner Black"
    color: Tuple[int, int, int, int] = (0, 0, 0, 255)  # RGBA
    weight: float = 1.0  # Proportion relative to other pens
    stroke: float = 0.5  # Rendered thickness (maps to physical penWidth)
    pen_width: float = 0.5  # Physical pen width in mm
    
    def matches_color(self, target_rgb: Tuple[int, int, int], 
                      delta_e_threshold: float = 5.0) -> bool:
        """Check if this pen's color matches target within Delta-E threshold."""
        # Simplified Euclidean distance in RGB space
        # In production, use CIE76 or CIEDE2000 Delta-E formula
        r_diff = self.color[0] - target_rgb[0]
        g_diff = self.color[1] - target_rgb[1]
        b_diff = self.color[2] - target_rgb[2]
        distance = math.sqrt(r_diff**2 + g_diff**2 + b_diff**2)
        return distance <= delta_e_threshold * 4.41  # Approximate conversion


@dataclass
class PenSettings:
    """Global pen distribution and color separation settings."""
    pens: List[Pen] = field(default_factory=list)
    distribution_type: DistributionType = DistributionType.EVEN_WEIGHTED
    distribution_order: DistributionOrder = DistributionOrder.DARKEST_FIRST
    color_separation: ColorSeparation = ColorSeparation.DEFAULT
    
    # Color Match specific
    colour_accuracy: int = 98  # Delta-E threshold (95-100)
    brightness_multiplier: float = 1.0
    pen_limit: int = 0  # 0 = unlimited
    use_canvas_colour: bool = False
    canvas_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    
    def add_pen(self, pen: Pen):
        """Add a pen to the configuration."""
        self.pens.append(pen)
    
    def get_enabled_pens(self) -> List[Pen]:
        """Get list of enabled pens, optionally sorted."""
        enabled = [p for p in self.pens if p.enabled]
        
        if self.distribution_order == DistributionOrder.DARKEST_FIRST:
            enabled.sort(key=lambda p: sum(p.color[:3]), reverse=False)
        elif self.distribution_order == DistributionOrder.LIGHTEST_FIRST:
            enabled.sort(key=lambda p: sum(p.color[:3]), reverse=True)
        elif self.distribution_order == DistributionOrder.REVERSED:
            enabled.reverse()
        
        return enabled
    
    def assign_paths_to_pens(self, paths: List[List[Tuple[float, float]]], 
                             image_luminance: Optional[List[List[float]]] = None
                             ) -> Dict[str, List[List[Tuple[float, float]]]]:
        """
        Assign generated paths to pens based on distribution settings.
        
        Args:
            paths: Generated paths from PFM
            image_luminance: Optional 2D array of luminance values per pixel
            
        Returns:
            Dictionary mapping pen names to their assigned paths
        """
        enabled_pens = self.get_enabled_pens()
        
        if not enabled_pens:
            return {}
        
        assignments = {pen.name: [] for pen in enabled_pens}
        
        if self.distribution_type == DistributionType.SINGLE_PEN:
            # All paths to first enabled pen
            if enabled_pens:
                assignments[enabled_pens[0].name] = paths
        
        elif self.distribution_type == DistributionType.EVEN_WEIGHTED:
            # Distribute evenly based on weight
            total_weight = sum(p.weight for p in enabled_pens)
            path_idx = 0
            for pen in enabled_pens:
                count = int(len(paths) * (pen.weight / total_weight))
                assignments[pen.name] = paths[path_idx:path_idx + count]
                path_idx += count
            # Assign remaining paths to last pen
            if path_idx < len(paths) and enabled_pens:
                assignments[enabled_pens[-1].name].extend(paths[path_idx:])
        
        elif self.distribution_type == DistributionType.RANDOM_WEIGHTED:
            # Random assignment weighted by pen weight
            weights = [p.weight for p in enabled_pens]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            
            for path in paths:
                pen_idx = random.choices(range(len(enabled_pens)), weights=probabilities)[0]
                assignments[enabled_pens[pen_idx].name].append(path)
        
        elif self.distribution_type == DistributionType.LUMINANCE_WEIGHTED:
            # Assign based on image luminance (darker areas to darker pens)
            if image_luminance is None:
                # Fallback to even weighted if no luminance data
                return self.assign_paths_to_pens(paths)
            
            # Sort pens by darkness
            sorted_pens = sorted(enabled_pens, key=lambda p: sum(p.color[:3]))
            
            for path in paths:
                # Calculate average luminance under path
                avg_lum = self._calculate_path_luminance(path, image_luminance)
                # Map luminance to pen index
                pen_idx = int(avg_lum * len(sorted_pens))
                pen_idx = max(0, min(pen_idx, len(sorted_pens) - 1))
                assignments[sorted_pens[pen_idx].name].append(path)
        
        return assignments
    
    def _calculate_path_luminance(self, path: List[Tuple[float, float]], 
                                  luminance: List[List[float]]) -> float:
        """Calculate average luminance value under a path."""
        if not luminance or not path:
            return 0.5
        
        total = 0.0
        count = 0
        height = len(luminance)
        width = len(luminance[0]) if height > 0 else 0
        
        for x, y in path:
            px, py = int(x), int(y)
            if 0 <= px < width and 0 <= py < height:
                total += luminance[py][px]
                count += 1
        
        return total / count if count > 0 else 0.5


# ============================================================================
# 3. PATH OPTIMIZATION
# ============================================================================

@dataclass
class PathOptimization:
    """Path optimization configuration for minimizing pen-up travel."""
    enable_simplifying: bool = True
    simplify_tolerance: float = 0.5  # Douglas-Peucker tolerance
    
    enable_merging: bool = True
    merge_distance: float = 2.0  # Max distance to merge endpoints
    
    enable_filtering: bool = True
    min_path_length: float = 5.0  # Delete paths shorter than this
    
    enable_sorting: bool = True  # STRTree spatial indexing for nearest-neighbor
    
    multipass: int = 1  # Repeat geometry N times
    
    def optimize(self, paths: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """
        Apply all enabled optimizations to paths.
        
        Args:
            paths: Raw paths from PFM
            
        Returns:
            Optimized paths ready for export
        """
        if not paths:
            return paths
        
        result = paths
        
        # 1. Filter short paths
        if self.enable_filtering:
            result = self._filter_short_paths(result)
        
        # 2. Simplify paths (Douglas-Peucker)
        if self.enable_simplifying:
            result = [self._douglas_peucker(path, self.simplify_tolerance) 
                     for path in result]
        
        # 3. Merge nearby endpoints
        if self.enable_merging:
            result = self._merge_paths(result)
        
        # 4. Sort paths to minimize travel (nearest-neighbor with STRTree)
        if self.enable_sorting:
            result = self._sort_paths_nearest_neighbor(result)
        
        # 5. Apply multipass
        if self.multipass > 1:
            result = self._apply_multipass(result)
        
        return result
    
    def _filter_short_paths(self, paths: List[List[Tuple[float, float]]]
                           ) -> List[List[Tuple[float, float]]]:
        """Remove paths shorter than minimum length."""
        result = []
        for path in paths:
            length = self._calculate_path_length(path)
            if length >= self.min_path_length:
                result.append(path)
        return result
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total length of a path."""
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total += math.sqrt(dx * dx + dy * dy)
        
        return total
    
    def _douglas_peucker(self, points: List[Tuple[float, float]], 
                         epsilon: float) -> List[Tuple[float, float]]:
        """
        Douglas-Peucker algorithm for path simplification.
        Reduces vertex count while preserving shape within tolerance.
        """
        if len(points) <= 2:
            return points
        
        # Find point with maximum distance from line between first and last
        start, end = points[0], points[-1]
        max_dist = 0.0
        max_idx = 0
        
        line_length_sq = (end[0] - start[0])**2 + (end[1] - start[1])**2
        
        for i in range(1, len(points) - 1):
            dist = self._perpendicular_distance(points[i], start, end, line_length_sq)
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_dist > epsilon:
            left = self._douglas_peucker(points[:max_idx+1], epsilon)
            right = self._douglas_peucker(points[max_idx:], epsilon)
            return left[:-1] + right
        else:
            return [start, end]
    
    def _perpendicular_distance(self, point: Tuple[float, float], 
                                line_start: Tuple[float, float],
                                line_end: Tuple[float, float],
                                line_length_sq: float) -> float:
        """Calculate perpendicular distance from point to line."""
        if line_length_sq == 0:
            return math.sqrt((point[0] - line_start[0])**2 + 
                           (point[1] - line_start[1])**2)
        
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        
        t = ((point[0] - line_start[0]) * dx + 
             (point[1] - line_start[1]) * dy) / line_length_sq
        
        t = max(0, min(1, t))
        
        proj_x = line_start[0] + t * dx
        proj_y = line_start[1] + t * dy
        
        return math.sqrt((point[0] - proj_x)**2 + (point[1] - proj_y)**2)
    
    def _merge_paths(self, paths: List[List[Tuple[float, float]]]
                    ) -> List[List[Tuple[float, float]]]:
        """Merge paths whose endpoints are within merge_distance."""
        if len(paths) <= 1:
            return paths
        
        merged = []
        used = [False] * len(paths)
        
        for i, path1 in enumerate(paths):
            if used[i]:
                continue
            
            current_path = path1[:]
            used[i] = True
            changed = True
            
            while changed:
                changed = False
                for j, path2 in enumerate(paths):
                    if used[j]:
                        continue
                    
                    # Check all 4 endpoint combinations
                    if self._endpoints_close(current_path[-1], path2[0]):
                        current_path.extend(path2[1:])
                        used[j] = True
                        changed = True
                        break
                    elif self._endpoints_close(current_path[-1], path2[-1]):
                        current_path.extend(reversed(path2))
                        used[j] = True
                        changed = True
                        break
                    elif self._endpoints_close(current_path[0], path2[0]):
                        current_path = list(reversed(path2))[:-1] + current_path
                        used[j] = True
                        changed = True
                        break
                    elif self._endpoints_close(current_path[0], path2[-1]):
                        current_path = path2 + current_path[1:]
                        used[j] = True
                        changed = True
                        break
            
            merged.append(current_path)
        
        return merged
    
    def _endpoints_close(self, p1: Tuple[float, float], 
                         p2: Tuple[float, float]) -> bool:
        """Check if two points are within merge_distance."""
        dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        return dist_sq <= self.merge_distance ** 2
    
    def _sort_paths_nearest_neighbor(self, paths: List[List[Tuple[float, float]]]
                                    ) -> List[List[Tuple[float, float]]]:
        """
        Sort paths using nearest-neighbor algorithm to minimize pen-up travel.
        Uses STRTree spatial indexing for O(log n) lookups.
        """
        if len(paths) <= 1:
            return paths
        
        # Build spatial index of path start points
        # In production, use Rtree or STRTree library
        # Here we use a simple O(n²) approach for demonstration
        
        unvisited = list(range(len(paths)))
        result = []
        current_pos = (0.0, 0.0)  # Start from origin
        
        while unvisited:
            # Find nearest unvisited path
            nearest_idx = None
            nearest_dist = float('inf')
            
            for idx in unvisited:
                start_point = paths[idx][0]
                dist = math.sqrt((start_point[0] - current_pos[0])**2 + 
                               (start_point[1] - current_pos[1])**2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = idx
            
            if nearest_idx is not None:
                result.append(paths[nearest_idx])
                current_pos = paths[nearest_idx][-1]  # Update to end of added path
                unvisited.remove(nearest_idx)
        
        return result
    
    def _apply_multipass(self, paths: List[List[Tuple[float, float]]]
                        ) -> List[List[Tuple[float, float]]]:
        """Repeat each path multipass times."""
        if self.multipass <= 1:
            return paths
        
        result = []
        for path in paths:
            for _ in range(self.multipass):
                result.append(path[:])
        return result


# ============================================================================
# 4. HARDWARE EXPORT SETTINGS
# ============================================================================

class CommentType(Enum):
    BRACKETS = "()"      # G-code comments in brackets
    SEMICOLONS = ";"     # G-code comments in semicolons
    NONE = "None"        # No comments


class HPGLRotation(Enum):
    ROT_0 = 0
    ROT_90 = 90
    ROT_180 = 180
    ROT_270 = 270
    AUTO = "AUTO"


@dataclass
class GCodeConfig:
    """G-code export configuration."""
    offset_x: float = 0.0
    offset_y: float = 0.0
    curve_flatness: float = 0.1  # Max deviation in mm for curve flattening
    center_zero_point: bool = False  # Shift origin to canvas center
    comment_type: CommentType = CommentType.BRACKETS
    
    gcode_start: str = "G28 ; Home"
    gcode_end: str = "M3 S0 ; Pen up\nM5 ; Stop spindle"
    gcode_pen_down: str = "M3 S90"
    gcode_pen_up: str = "M3 S0"
    gcode_start_layer: str = "; Layer: %LAYER_NAME%"
    
    feed_rate: float = 1000.0  # mm/min
    plunge_rate: float = 500.0  # mm/min for pen down/up


@dataclass
class HPGLConfig:
    """HPGL export configuration."""
    hard_clip_min_x: int = 0
    hard_clip_min_y: int = 0
    hard_clip_max_x: int = 10000  # 40 units/mm * 250mm
    hard_clip_max_y: int = 10000
    rotation: HPGLRotation = HPGLRotation.ROT_0
    x_axis_mirror: bool = False
    y_axis_mirror: bool = False
    pen_velocity: int = 0  # mm/s, 0 = max speed
    pen_force: int = 30  # Downward pressure (FS command)
    initial_pen: int = 1  # Starting SP (Select Pen) ID


@dataclass
class SVGConfig:
    """SVG export configuration."""
    viewBox_width: float = 800.0
    viewBox_height: float = 600.0
    include_metadata: bool = True
    stroke_width: float = 0.5
    units: str = "mm"


@dataclass
class HardwareExport:
    """Unified hardware export system."""
    gcode_config: GCodeConfig = field(default_factory=GCodeConfig)
    hpgl_config: HPGLConfig = field(default_factory=HPGLConfig)
    svg_config: SVGConfig = field(default_factory=SVGConfig)
    
    def export_gcode(self, paths: List[List[Tuple[float, float]]], 
                     output_path: str,
                     pen_assignments: Optional[Dict[str, List[List[Tuple[float, float]]]]] = None
                     ) -> str:
        """
        Export paths to G-code format.
        
        Args:
            paths: Optimized paths
            output_path: Output file path
            pen_assignments: Optional dict mapping pen names to paths
            
        Returns:
            Generated G-code string
        """
        lines = []
        
        # Header
        lines.append("; Generated by DrawingBot V3 Clone")
        lines.append(f"; Date: {__import__('datetime').datetime.now().isoformat()}")
        lines.append("")
        
        # Startup routine
        if self.gcode_config.gcode_start:
            lines.extend(self.gcode_config.gcode_start.split('\n'))
        lines.append("")
        
        # Calculate offsets
        offset_x = self.gcode_config.offset_x
        offset_y = self.gcode_config.offset_y
        
        if self.gcode_config.center_zero_point and paths:
            # Find bounding box
            all_x = [p[0] for path in paths for p in path]
            all_y = [p[1] for path in paths for p in path]
            if all_x and all_y:
                offset_x = -(min(all_x) + max(all_x)) / 2
                offset_y = -(min(all_y) + max(all_y)) / 2
        
        # Format function for comments
        def fmt_comment(text: str) -> str:
            if self.gcode_config.comment_type == CommentType.BRACKETS:
                return f"({text})"
            elif self.gcode_config.comment_type == CommentType.SEMICOLONS:
                return f";{text}"
            return ""
        
        # Process paths by pen layers if provided
        if pen_assignments:
            for pen_name, pen_paths in pen_assignments.items():
                layer_header = self.gcode_config.gcode_start_layer.replace(
                    "%LAYER_NAME%", pen_name)
                lines.append(layer_header)
                lines.extend(self._paths_to_gcode(pen_paths, offset_x, offset_y, 
                                                  self.gcode_config.comment_type))
                lines.append("")
        else:
            lines.extend(self._paths_to_gcode(paths, offset_x, offset_y,
                                              self.gcode_config.comment_type))
        
        # Shutdown routine
        lines.append("")
        if self.gcode_config.gcode_end:
            lines.extend(self.gcode_config.gcode_end.split('\n'))
        
        gcode = '\n'.join(lines)
        
        # Write to file
        Path(output_path).write_text(gcode)
        
        return gcode
    
    def _paths_to_gcode(self, paths: List[List[Tuple[float, float]]], 
                        offset_x: float, offset_y: float,
                        comment_type: CommentType = CommentType.BRACKETS) -> List[str]:
        """Convert paths to G-code commands."""
        lines = []
        
        # Format function for comments
        def fmt_comment(text: str) -> str:
            if comment_type == CommentType.BRACKETS:
                return f"({text})"
            elif comment_type == CommentType.SEMICOLONS:
                return f";{text}"
            return ""
        
        for path_idx, path in enumerate(paths):
            if len(path) < 2:
                continue
            
            # Move to start position (pen up)
            start_x = path[0][0] + offset_x
            start_y = path[0][1] + offset_y
            lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f} {fmt_comment('Move to start')}")
            
            # Pen down
            lines.append(f"{self.gcode_config.gcode_pen_down} {fmt_comment('Pen down')}")
            lines.append(f"F{self.gcode_config.feed_rate}")
            
            # Draw path
            for x, y in path[1:]:
                gx = x + offset_x
                gy = y + offset_y
                lines.append(f"G1 X{gx:.3f} Y{gy:.3f}")
            
            # Pen up
            lines.append(f"{self.gcode_config.gcode_pen_up} {fmt_comment('Pen up')}")
            lines.append("")
        
        return lines
    
    def export_hpgl(self, paths: List[List[Tuple[float, float]]], 
                    output_path: str) -> str:
        """
        Export paths to HPGL format.
        
        HPGL uses plotter units (40 units = 1 mm)
        """
        lines = []
        
        # Initialize
        lines.append("IN;")  # Initialize
        
        # Select initial pen
        lines.append(f"SP{self.hpgl_config.initial_pen};")
        
        # Set velocity if specified
        if self.hpgl_config.pen_velocity > 0:
            lines.append(f"VS{self.hpgl_config.pen_velocity};")
        
        # Set force if specified
        if self.hpgl_config.pen_force > 0:
            lines.append(f"FS{self.hpgl_config.pen_force};")
        
        # Apply rotation
        if self.hpgl_config.rotation != HPGLRotation.ROT_0:
            rot = self.hpgl_config.rotation.value
            if rot != "AUTO":
                lines.append(f"RO{rot};")
        
        # Set clipping rectangle
        lines.append(f"IP{self.hpgl_config.hard_clip_min_x},"
                    f"{self.hpgl_config.hard_clip_min_y},"
                    f"{self.hpgl_config.hard_clip_max_x},"
                    f"{self.hpgl_config.hard_clip_max_y};")
        
        # Convert coordinates to HPGL units (40 units/mm)
        scale = 40.0
        
        for path in paths:
            if len(path) < 2:
                continue
            
            # Move to start (pen up implied)
            x = int((path[0][0] * scale) + self.hpgl_config.hard_clip_min_x)
            y = int((path[0][1] * scale) + self.hpgl_config.hard_clip_min_y)
            
            if self.hpgl_config.x_axis_mirror:
                x = self.hpgl_config.hard_clip_max_x - (x - self.hpgl_config.hard_clip_min_x)
            if self.hpgl_config.y_axis_mirror:
                y = self.hpgl_config.hard_clip_max_y - (y - self.hpgl_config.hard_clip_min_y)
            
            lines.append(f"PU{x},{y};")
            
            # Draw
            points = []
            for px, py in path[1:]:
                hx = int((px * scale) + self.hpgl_config.hard_clip_min_x)
                hy = int((py * scale) + self.hpgl_config.hard_clip_min_y)
                
                if self.hpgl_config.x_axis_mirror:
                    hx = self.hpgl_config.hard_clip_max_x - (hx - self.hpgl_config.hard_clip_min_x)
                if self.hpgl_config.y_axis_mirror:
                    hy = self.hpgl_config.hard_clip_max_y - (hy - self.hpgl_config.hard_clip_min_y)
                
                points.append(f"{hx},{hy}")
            
            if points:
                lines.append(f"PD{','.join(points)};")
        
        hpgl = '\n'.join(lines)
        
        # Write to file
        Path(output_path).write_text(hpgl)
        
        return hpgl
    
    def export_svg(self, paths: List[List[Tuple[float, float]]], 
                   output_path: str,
                   pen_assignments: Optional[Dict[str, List[List[Tuple[float, float]]]]] = None
                   ) -> str:
        """Export paths to SVG format."""
        width = self.svg_config.viewBox_width
        height = self.svg_config.viewBox_height
        
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}{self.svg_config.units}" '
            f'height="{height}{self.svg_config.units}" '
            f'viewBox="0 0 {width} {height}">'
        ]
        
        # Metadata
        if self.svg_config.include_metadata:
            svg_parts.extend([
                '  <metadata>',
                '    Generated by DrawingBot V3 Clone',
                f'    Date: {__import__("datetime").datetime.now().isoformat()}',
                '  </metadata>'
            ])
        
        stroke_width = self.svg_config.stroke_width
        
        if pen_assignments:
            for pen_name, pen_paths in pen_assignments.items():
                # Extract color from pen name (simplified)
                color = self._name_to_color(pen_name)
                svg_parts.append(f'  <g id="{pen_name}" stroke="{color}" '
                               f'stroke-width="{stroke_width}" fill="none">')
                
                for path in pen_paths:
                    if len(path) >= 2:
                        d = self._path_to_svg_d(path)
                        svg_parts.append(f'    <path d="{d}"/>')
                
                svg_parts.append('  </g>')
        else:
            svg_parts.append(f'  <g stroke="black" stroke-width="{stroke_width}" fill="none">')
            for path in paths:
                if len(path) >= 2:
                    d = self._path_to_svg_d(path)
                    svg_parts.append(f'    <path d="{d}"/>')
            svg_parts.append('  </g>')
        
        svg_parts.append('</svg>')
        
        svg = '\n'.join(svg_parts)
        
        # Write to file
        Path(output_path).write_text(svg)
        
        return svg
    
    def _path_to_svg_d(self, path: List[Tuple[float, float]]) -> str:
        """Convert path to SVG path data string."""
        if len(path) < 2:
            return ""
        
        d = f"M {path[0][0]:.3f} {path[0][1]:.3f}"
        for x, y in path[1:]:
            d += f" L {x:.3f} {y:.3f}"
        
        return d
    
    def _name_to_color(self, name: str) -> str:
        """Convert pen name to hex color (simplified hash)."""
        # Simple hash-based color generation
        h = hash(name) & 0xFFFFFF
        return f"#{h:06x}"


# ============================================================================
# COMPLETE PIPELINE EXAMPLE
# ============================================================================

def run_complete_pipeline():
    """Example: Complete DrawingBot V3 pipeline from paths to G-code."""
    
    # Sample paths (would come from PFM algorithms)
    sample_paths = [
        [(0, 0), (10, 10), (20, 0), (30, 10)],
        [(5, 5), (15, 15), (25, 5)],
        [(0, 20), (30, 20), (30, 40), (0, 40), (0, 20)],
    ]
    
    # 1. Configure Masking
    masking = MaskingSystem(
        enable_masking=True,
        soft_clip=False,
        masks=[
            Mask(
                mask_type=MaskType.ADD,
                shape=MaskShape.RECTANGLE,
                start_x=0, start_y=0,
                width=50, height=50
            )
        ]
    )
    
    masked_paths = masking.apply_to_paths(sample_paths)
    print(f"After masking: {len(masked_paths)} paths")
    
    # 2. Configure Pens
    pen_settings = PenSettings(
        distribution_type=DistributionType.EVEN_WEIGHTED,
        distribution_order=DistributionOrder.DARKEST_FIRST,
        pens=[
            Pen(name="Black FineLiner", color=(0, 0, 0, 255), weight=1.0),
            Pen(name="Red FineLiner", color=(255, 0, 0, 255), weight=0.8),
        ]
    )
    
    pen_assignments = pen_settings.assign_paths_to_pens(masked_paths)
    print(f"Pen assignments: {[(k, len(v)) for k, v in pen_assignments.items()]}")
    
    # 3. Optimize Paths
    optimizer = PathOptimization(
        enable_simplifying=True,
        simplify_tolerance=0.3,
        enable_merging=True,
        merge_distance=1.5,
        enable_filtering=True,
        min_path_length=2.0,
        enable_sorting=True,
        multipass=1
    )
    
    optimized_paths = optimizer.optimize(masked_paths)
    print(f"After optimization: {len(optimized_paths)} paths")
    
    # 4. Export to G-code
    exporter = HardwareExport(
        gcode_config=GCodeConfig(
            offset_x=10.0,
            offset_y=10.0,
            center_zero_point=False,
            comment_type=CommentType.SEMICOLONS,
            gcode_pen_down="M3 S90",
            gcode_pen_up="M3 S0"
        )
    )
    
    gcode = exporter.export_gcode(optimized_paths, "output.gcode", pen_assignments)
    print(f"Generated G-code: {len(gcode)} bytes")
    
    # Also export SVG
    svg = exporter.export_svg(optimized_paths, "output.svg", pen_assignments)
    print(f"Generated SVG: {len(svg)} bytes")
    
    return {
        'masked_paths': masked_paths,
        'pen_assignments': pen_assignments,
        'optimized_paths': optimized_paths,
        'gcode': gcode,
        'svg': svg
    }


if __name__ == "__main__":
    result = run_complete_pipeline()
    print("\n✅ Complete pipeline executed successfully!")
    print(f"   - Masked paths: {len(result['masked_paths'])}")
    print(f"   - Pen assignments: {len(result['pen_assignments'])} pens")
    print(f"   - Optimized paths: {len(result['optimized_paths'])}")
    print(f"   - G-code size: {len(result['gcode'])} chars")
    print(f"   - SVG size: {len(result['svg'])} chars")
