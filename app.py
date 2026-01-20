from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
import os
import cv2

app = Flask(__name__)

# Environment variables
API_KEY = os.environ.get('API_KEY', None)
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE_MB', 10)) * 1024 * 1024
DEFAULT_THRESHOLD = int(os.environ.get('DEFAULT_THRESHOLD', 100))
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def verify_api_key():
    """Verify API key if set in environment"""
    if API_KEY:
        provided_key = request.headers.get('X-API-Key') or request.form.get('api_key')
        if provided_key != API_KEY:
            return jsonify({
                "error": "Unauthorized",
                "message": "Invalid or missing API key"
            }), 401
    return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "Image Color Replacer API",
        "status": "running",
        "version": "2.1.0",
        "authentication": "required" if API_KEY else "not required",
        "endpoints": {
            "/health": "GET - Health check",
            "/admin": "GET - Admin panel (for keep-alive)",
            "/check-dependencies": "GET - Check installed packages",
            "/replace-dark-to-white": "POST - Replace dark colors with white",
            "/replace-light-to-dark": "POST - Replace light colors with dark",
            "/replace-to-color": "POST - Replace dark/light to any color (RECOMMENDED)",
            "/smart-color-replace": "POST - Auto-detect and replace solid colors",
            "/force-solid-black": "POST - Force entire logo to solid black ⭐",
            "/force-solid-white": "POST - Force entire logo to solid white ⭐",
            "/invert-colors": "POST - Invert all colors"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "Service is running"})

@app.route('/check-dependencies', methods=['GET'])
def check_dependencies():
    """Check if all dependencies are available"""
    deps = {}
    
    try:
        import flask
        deps['flask'] = flask.__version__
    except:
        deps['flask'] = 'NOT INSTALLED'
    
    try:
        import PIL
        deps['pillow'] = PIL.__version__
    except:
        deps['pillow'] = 'NOT INSTALLED'
    
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except:
        deps['numpy'] = 'NOT INSTALLED'
    
    try:
        import cv2
        deps['opencv'] = cv2.__version__
    except Exception as e:
        deps['opencv'] = f'NOT INSTALLED: {str(e)}'
    
    return jsonify({
        "dependencies": deps,
        "opencv_available": 'cv2' in dir()
    })

@app.route('/admin', methods=['GET'])
def admin():
    """Fun admin endpoint for keep-alive pings"""
    import random
    from datetime import datetime
    
    funny_messages = [
        "🎨 Image processor standing by, captain!",
        "🖼️ Still here, converting pixels like a boss!",
        "🎭 Awake and ready to make colors dance!",
        "🚀 Service online! No images harmed in the making of this response.",
        "🎪 The pixel circus is open for business!"
    ]
    
    return jsonify({
        "status": "alive_and_kicking",
        "message": random.choice(funny_messages),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "coffee_level": f"{random.randint(60, 100)}%"
    })

@app.route('/smart-print-ready', methods=['POST'])
def smart_print_ready():
    """
    🎯 ONE ENDPOINT TO RULE THEM ALL
    
    Automatically analyzes logo and creates perfect print-ready version.
    Just send image + print_color, get optimal result.
    
    Parameters:
    - image: file (required)
    - print_color: 'white' or 'black' (required)
    - mode: 'auto', 'bordered', 'flat' (default: 'auto')
    
    The endpoint automatically:
    - Detects and removes background
    - Identifies borders vs fills
    - Preserves logo structure
    - Optimizes for print/mockup use
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        print_color = request.form.get('print_color', '').lower()
        mode = request.form.get('mode', 'auto').lower()
        
        if print_color not in ['white', 'black']:
            return jsonify({
                "error": "print_color is required and must be 'white' or 'black'",
                "example": "print_color=white (for dark shirts) or print_color=black (for light shirts)"
            }), 400
        
        # Load image
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        height, width = data.shape[:2]
        total_pixels = height * width
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        original_alpha = a.copy()
        
        # ============================================================
        # STEP 1: ANALYZE
        # ============================================================
        
        analysis = {
            "background_type": "unknown",
            "has_borders": False,
            "logo_type": "unknown",
            "complexity": "simple",
            "decisions": []
        }
        
        # 1a. Analyze transparency
        has_transparency = np.any(a < 255)
        fully_transparent_pixels = np.sum(a == 0)
        transparency_ratio = fully_transparent_pixels / total_pixels
        
        # 1b. Analyze background from corners and edges
        corner_samples = [
            data[0, 0, :3],
            data[0, -1, :3],
            data[-1, 0, :3],
            data[-1, -1, :3],
            data[0, width//2, :3],
            data[-1, width//2, :3],
            data[height//2, 0, :3],
            data[height//2, -1, :3]
        ]
        avg_corner = np.mean(corner_samples, axis=0)
        corner_std = np.std(corner_samples, axis=0)
        
        # Determine background type
        if transparency_ratio > 0.1:
            analysis["background_type"] = "transparent"
        elif np.mean(avg_corner) > 230 and np.mean(corner_std) < 20:
            analysis["background_type"] = "white"
        elif np.mean(avg_corner) < 25 and np.mean(corner_std) < 20:
            analysis["background_type"] = "black"
        else:
            analysis["background_type"] = "colored_or_complex"
        
        # 1c. Analyze edges/borders
        gray = cv2.cvtColor(data[:,:,:3], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixel_count = np.sum(edges > 0)
        edge_ratio = edge_pixel_count / total_pixels
        
        analysis["has_borders"] = edge_ratio > 0.005  # More than 0.5% edges
        
        # 1d. Analyze color distribution
        luminance = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        
        non_transparent_mask = a > 0
        if np.any(non_transparent_mask):
            lum_values = luminance[non_transparent_mask]
            dark_ratio = np.sum(lum_values < 60) / len(lum_values)
            light_ratio = np.sum(lum_values > 200) / len(lum_values)
            mid_ratio = 1 - dark_ratio - light_ratio
            
            # Check color variance (is it multi-colored?)
            color_std = np.std(data[non_transparent_mask][:, :3], axis=0)
            is_multicolor = np.mean(color_std) > 50
        else:
            dark_ratio = light_ratio = mid_ratio = 0
            is_multicolor = False
        
        # 1e. Determine logo type
        if mode != 'auto':
            analysis["logo_type"] = mode
        elif not analysis["has_borders"] and not is_multicolor:
            analysis["logo_type"] = "flat"
        elif analysis["has_borders"]:
            analysis["logo_type"] = "bordered"
        else:
            analysis["logo_type"] = "complex"
        
        analysis["complexity"] = "complex" if is_multicolor else "simple"
        
        # ============================================================
        # STEP 2: DECIDE PROCESSING STRATEGY
        # ============================================================
        
        # 2a. Background removal strategy
        remove_bg = analysis["background_type"] in ["white", "black", "colored_or_complex"]
        if remove_bg:
            analysis["decisions"].append(f"Remove {analysis['background_type']} background")
        
        # 2b. Border handling strategy
        if analysis["logo_type"] == "flat":
            border_strategy = "none"
            analysis["decisions"].append("Flat logo - simple solid conversion")
        elif analysis["logo_type"] == "bordered":
            border_strategy = "transparent"
            analysis["decisions"].append("Bordered logo - make borders transparent for shirt color to show")
        else:
            border_strategy = "smart"
            analysis["decisions"].append("Complex logo - smart edge preservation")
        
        # 2c. Sensitivity based on complexity
        if analysis["complexity"] == "complex":
            canny_low, canny_high = 30, 100
            dilate_size = 2
            analysis["decisions"].append("High sensitivity for complex logo")
        else:
            canny_low, canny_high = 50, 150
            dilate_size = 2
            analysis["decisions"].append("Medium sensitivity for simple logo")
        
        # ============================================================
        # STEP 3: PROCESS
        # ============================================================
        
        # 3a. Remove background
        if remove_bg:
            if analysis["background_type"] == "white":
                bg_mask = (r > 235) & (g > 235) & (b > 235)
            elif analysis["background_type"] == "black":
                bg_mask = (r < 20) & (g < 20) & (b < 20)
            else:
                # Match corner color with tolerance
                diff = np.sqrt(
                    (r.astype(np.float32) - avg_corner[0])**2 +
                    (g.astype(np.float32) - avg_corner[1])**2 +
                    (b.astype(np.float32) - avg_corner[2])**2
                )
                bg_mask = diff < 40
            
            # Refine background mask - remove small holes
            bg_mask_uint8 = bg_mask.astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            bg_mask_uint8 = cv2.morphologyEx(bg_mask_uint8, cv2.MORPH_CLOSE, kernel)
            bg_mask = bg_mask_uint8 > 127
        else:
            bg_mask = original_alpha == 0
        
        # Include already transparent pixels in background
        bg_mask = bg_mask | (original_alpha == 0)
        
        # 3b. Detect and handle borders
        if border_strategy != "none":
            # Edge detection
            edges_detected = cv2.Canny(gray, canny_low, canny_high)
            
            # Dilate edges slightly
            kernel = np.ones((dilate_size, dilate_size), np.uint8)
            edges_dilated = cv2.dilate(edges_detected, kernel, iterations=1)
            
            # Dark pixels are likely borders/outlines
            dark_mask = luminance < 50
            
            # Combine: edges OR very dark pixels, but NOT background
            border_mask = ((edges_dilated > 0) | dark_mask) & ~bg_mask
            
            # Clean up border mask
            border_mask_uint8 = border_mask.astype(np.uint8) * 255
            border_mask_uint8 = cv2.morphologyEx(border_mask_uint8, cv2.MORPH_OPEN, kernel)
            border_mask = border_mask_uint8 > 127
        else:
            border_mask = np.zeros((height, width), dtype=bool)
        
        # 3c. Create fill mask
        fill_mask = ~bg_mask & ~border_mask
        
        # 3d. Build result image
        result = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Set colors
        if print_color == 'white':
            fill_rgba = [255, 255, 255, 255]
        else:
            fill_rgba = [0, 0, 0, 255]
        
        # Apply fills
        result[fill_mask] = fill_rgba
        
        # Apply borders based on strategy
        if border_strategy == "transparent":
            # Borders become transparent - shirt color shows through
            result[border_mask] = [0, 0, 0, 0]
        elif border_strategy == "smart":
            # For complex logos, keep some border definition
            # Use semi-transparency
            result[border_mask] = [0, 0, 0, 0]
        
        # Background is transparent
        result[bg_mask] = [0, 0, 0, 0]
        
        # 3e. Anti-aliasing preservation
        # Find edges of fill areas and smooth them
        fill_mask_uint8 = fill_mask.astype(np.uint8) * 255
        fill_edges = cv2.Canny(fill_mask_uint8, 100, 200)
        fill_edges_dilated = cv2.dilate(fill_edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # Apply slight transparency on edges for smoother appearance
        aa_mask = (fill_edges_dilated > 0) & fill_mask
        result[aa_mask, 3] = 220  # Slightly transparent edges
        
        # ============================================================
        # STEP 4: OUTPUT
        # ============================================================
        
        result_img = Image.fromarray(result, 'RGBA')
        
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=True)
        output.seek(0)
        
        # Stats
        fill_pixels = int(np.sum(fill_mask))
        border_pixels = int(np.sum(border_mask))
        bg_pixels = int(np.sum(bg_mask))
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'print_ready_{print_color}.png'
        )
        
        # Include analysis in headers for debugging
        response.headers['X-Print-Color'] = print_color.upper()
        response.headers['X-Background-Detected'] = analysis["background_type"]
        response.headers['X-Logo-Type'] = analysis["logo_type"]
        response.headers['X-Has-Borders'] = str(analysis["has_borders"])
        response.headers['X-Complexity'] = analysis["complexity"]
        response.headers['X-Fill-Pixels'] = str(fill_pixels)
        response.headers['X-Border-Pixels'] = str(border_pixels)
        response.headers['X-BG-Pixels-Removed'] = str(bg_pixels)
        response.headers['X-Decisions'] = "; ".join(analysis["decisions"])
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/smart-print-ready/analyze', methods=['POST'])
def smart_print_ready_analyze():
    """
    Debug endpoint - shows what smart-print-ready will detect
    Same analysis without processing, useful for debugging
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        height, width = data.shape[:2]
        total_pixels = height * width
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        
        # Transparency analysis
        has_transparency = np.any(a < 255)
        fully_transparent = int(np.sum(a == 0))
        
        # Background analysis
        corner_samples = [
            data[0, 0, :3], data[0, -1, :3],
            data[-1, 0, :3], data[-1, -1, :3]
        ]
        avg_corner = np.mean(corner_samples, axis=0)
        
        if fully_transparent / total_pixels > 0.1:
            bg_type = "transparent"
        elif np.mean(avg_corner) > 230:
            bg_type = "white"
        elif np.mean(avg_corner) < 25:
            bg_type = "black"
        else:
            bg_type = "colored_or_complex"
        
        # Edge analysis
        gray = cv2.cvtColor(data[:,:,:3], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = int(np.sum(edges > 0))
        has_borders = edge_pixels / total_pixels > 0.005
        
        # Color analysis
        non_transparent = a > 0
        if np.any(non_transparent):
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            lum_values = luminance[non_transparent]
            
            dark_pixels = int(np.sum(lum_values < 60))
            mid_pixels = int(np.sum((lum_values >= 60) & (lum_values <= 200)))
            light_pixels = int(np.sum(lum_values > 200))
            
            color_std = np.std(data[non_transparent][:, :3], axis=0)
            is_multicolor = bool(np.mean(color_std) > 50)
        else:
            dark_pixels = mid_pixels = light_pixels = 0
            is_multicolor = False
        
        # Determine logo type
        if not has_borders and not is_multicolor:
            logo_type = "flat"
        elif has_borders:
            logo_type = "bordered"
        else:
            logo_type = "complex"
        
        return jsonify({
            "image": {
                "dimensions": f"{width}x{height}",
                "total_pixels": total_pixels
            },
            "background": {
                "type": bg_type,
                "has_transparency": has_transparency,
                "transparent_pixels": fully_transparent,
                "corner_avg_color": [int(c) for c in avg_corner],
                "will_be_removed": bg_type in ["white", "black", "colored_or_complex"]
            },
            "structure": {
                "edge_pixels": edge_pixels,
                "edge_ratio": f"{(edge_pixels/total_pixels)*100:.2f}%",
                "has_borders": has_borders,
                "logo_type": logo_type
            },
            "colors": {
                "dark_pixels": dark_pixels,
                "mid_pixels": mid_pixels,
                "light_pixels": light_pixels,
                "is_multicolor": is_multicolor,
                "complexity": "complex" if is_multicolor else "simple"
            },
            "recommendations": {
                "for_dark_shirt": "POST /smart-print-ready with print_color=white",
                "for_light_shirt": "POST /smart-print-ready with print_color=black"
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Analysis failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500
        
@app.route('/force-solid-with-outline', methods=['POST'])
def force_solid_with_outline():
    """
    Convert logo to solid color while keeping it readable
    Adds outline/stroke to preserve shape definition
    
    Parameters:
    - image: file (required)
    - fill_color: 'black' or 'white' (default: 'black')
    - outline_color: 'white' or 'black' (default: opposite of fill)
    - outline_width: integer pixels (default: 3)
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # Get parameters
        fill_color = request.form.get('fill_color', 'black').lower()
        outline_color = request.form.get('outline_color', None)
        outline_width = int(request.form.get('outline_width', 3))
        
        # Auto-set outline color to contrast with fill
        if outline_color is None:
            outline_color = 'white' if fill_color == 'black' else 'black'
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        
        # Get original alpha channel (shape of logo)
        original_alpha = img.split()[3]
        
        # Create solid color version
        if fill_color == 'white':
            solid_fill = Image.new('RGB', img.size, (255, 255, 255))
        else:
            solid_fill = Image.new('RGB', img.size, (0, 0, 0))
        
        # Create outline by dilating the alpha channel
        # This makes the shape slightly bigger
        alpha_array = np.array(original_alpha)
        
        # Use morphological dilation to create outline
        kernel = np.ones((outline_width * 2 + 1, outline_width * 2 + 1), np.uint8)
        dilated = cv2.dilate(alpha_array, kernel, iterations=1)
        
        # The outline is the difference between dilated and original
        outline_mask = (dilated > 0) & (alpha_array == 0)
        
        # Create outline image
        if outline_color == 'white':
            outline_img = Image.new('RGBA', img.size, (255, 255, 255, 0))
        else:
            outline_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        
        outline_data = np.array(outline_img)
        if outline_color == 'white':
            outline_data[outline_mask] = [255, 255, 255, 255]
        else:
            outline_data[outline_mask] = [0, 0, 0, 255]
        
        outline_img = Image.fromarray(outline_data, 'RGBA')
        
        # Composite: outline first, then solid fill on top
        result = Image.new('RGBA', img.size, (0, 0, 0, 0))
        result.paste(outline_img, (0, 0), outline_img)
        result.paste(solid_fill, (0, 0), original_alpha)
        
        # Save
        output = BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'solid_{fill_color}_with_outline.png'
        )
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/force-solid-readable', methods=['POST'])
def force_solid_readable():
    """
    Convert logo to solid color while keeping ALL details readable
    Uses luminance-based edge preservation for better results
    
    Parameters:
    - image: file (required)
    - base_color: 'black' or 'white' (default: 'white')
    - edge_strength: 1-10 (default: 3, higher=thicker edges)
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        base_color = request.form.get('base_color', 'white').lower()
        edge_strength = int(request.form.get('edge_strength', 3))
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        rgb = np.array(img.convert('RGB'))
        alpha = np.array(img.split()[3])
        
        # Convert to grayscale to detect luminance differences
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Use Sobel edge detection (better for internal lines)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        sobel_edges = (sobel_edges / sobel_edges.max() * 255).astype(np.uint8)
        _, edge_mask = cv2.threshold(sobel_edges, 20, 255, cv2.THRESH_BINARY)
        
        # Thicken edges based on strength parameter
        if edge_strength > 1:
            kernel = np.ones((edge_strength, edge_strength), np.uint8)
            edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
        
        # Create result
        result = np.zeros((img.height, img.width, 4), dtype=np.uint8)
        
        # Set base color for all non-transparent pixels
        non_transparent = alpha > 0
        
        if base_color == 'white':
            result[non_transparent, 0:3] = [255, 255, 255]  # White base
            # Make edges black
            edge_pixels = (edge_mask > 0) & non_transparent
            result[edge_pixels, 0:3] = [0, 0, 0]
        else:
            result[non_transparent, 0:3] = [0, 0, 0]  # Black base
            # Make edges white
            edge_pixels = (edge_mask > 0) & non_transparent
            result[edge_pixels, 0:3] = [255, 255, 255]
        
        # Apply original alpha
        result[:, :, 3] = alpha
        
        # Create final image
        result_img = Image.fromarray(result, 'RGBA')
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        # Debug info
        edge_pixel_count = np.sum(edge_pixels)
        total_opaque = np.sum(non_transparent)
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'readable_{base_color}.png'
        )
        
        response.headers['X-Edge-Pixels'] = str(edge_pixel_count)
        response.headers['X-Total-Opaque-Pixels'] = str(total_opaque)
        response.headers['X-Edge-Percentage'] = f"{(edge_pixel_count/total_opaque*100):.2f}%"
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/smart-solid-conversion', methods=['POST'])
def smart_solid_conversion():
    """
    Intelligently convert multi-color logo to solid color
    Darker colors become strokes, lighter colors become fill
    
    Parameters:
    - image: file (required)
    - output_color: 'black' or 'white' (default: 'white')
    - invert_logic: 'true' or 'false' (default: 'false')
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        output_color = request.form.get('output_color', 'white').lower()
        invert_logic = request.form.get('invert_logic', 'false').lower() == 'true'
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        rgb = np.array(img.convert('RGB'))
        alpha = np.array(img.split()[3])
        
        # Calculate luminance (brightness) for each pixel
        luminance = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        
        # Non-transparent pixels only
        non_transparent = alpha > 0
        
        # Find threshold - use mean luminance of non-transparent pixels
        if np.any(non_transparent):
            threshold = np.mean(luminance[non_transparent])
        else:
            threshold = 127
        
        # Classify pixels as "dark" (strokes) or "light" (fill)
        if invert_logic:
            is_stroke = (luminance > threshold) & non_transparent
        else:
            is_stroke = (luminance < threshold) & non_transparent
        
        # Create result
        result = np.zeros((img.height, img.width, 4), dtype=np.uint8)
        
        if output_color == 'white':
            # White fill, black strokes
            result[non_transparent, 0:3] = [255, 255, 255]
            result[is_stroke, 0:3] = [0, 0, 0]
        else:
            # Black fill, white strokes
            result[non_transparent, 0:3] = [0, 0, 0]
            result[is_stroke, 0:3] = [255, 255, 255]
        
        # Apply alpha
        result[:, :, 3] = alpha
        
        result_img = Image.fromarray(result, 'RGBA')
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        stroke_pixels = np.sum(is_stroke)
        fill_pixels = np.sum(non_transparent) - stroke_pixels
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'smart_{output_color}.png'
        )
        
        response.headers['X-Stroke-Pixels'] = str(stroke_pixels)
        response.headers['X-Fill-Pixels'] = str(fill_pixels)
        response.headers['X-Luminance-Threshold'] = f"{threshold:.2f}"
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500
        
@app.route('/force-solid-black', methods=['POST'])
def force_solid_black():
    """
    Convert ANY logo to solid black while PRESERVING transparency
    Ultra-safe version that guarantees transparent pixels stay transparent
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Open image - force RGBA mode
        img = Image.open(file.stream)
        
        # Ensure we have an alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get the data as numpy array
        data = np.array(img, dtype=np.uint8)
        
        # Split into separate channels
        r = data[:, :, 0]
        g = data[:, :, 1]
        b = data[:, :, 2]
        alpha = data[:, :, 3]
        
        # Create new array with same shape
        result = np.zeros_like(data)
        
        # Set RGB to black for all pixels
        result[:, :, 0] = 0  # R
        result[:, :, 1] = 0  # G
        result[:, :, 2] = 0  # B
        
        # CRITICAL: Copy original alpha channel exactly as-is
        result[:, :, 3] = alpha
        
        # Create image from array
        result_img = Image.fromarray(result, mode='RGBA')
        
        # Save as PNG with transparency
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=False)
        output.seek(0)
        
        # Stats
        total_pixels = alpha.size
        transparent_pixels = np.sum(alpha == 0)
        semi_transparent = np.sum((alpha > 0) & (alpha < 255))
        fully_opaque = np.sum(alpha == 255)
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='solid_black.png'
        )
        
        response.headers['X-Total-Pixels'] = str(total_pixels)
        response.headers['X-Transparent-Pixels'] = str(transparent_pixels)
        response.headers['X-Semi-Transparent-Pixels'] = str(semi_transparent)
        response.headers['X-Opaque-Pixels'] = str(fully_opaque)
        response.headers['X-Output-Color'] = 'BLACK'
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/force-solid-white', methods=['POST'])
def force_solid_white():
    """
    Convert ANY logo to solid white while PRESERVING transparency
    Ultra-safe version that guarantees transparent pixels stay transparent
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Open image - force RGBA mode
        img = Image.open(file.stream)
        
        # Ensure we have an alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get the data as numpy array
        data = np.array(img, dtype=np.uint8)
        
        # Split into separate channels
        r = data[:, :, 0]
        g = data[:, :, 1]
        b = data[:, :, 2]
        alpha = data[:, :, 3]
        
        # Create new array with same shape
        result = np.zeros_like(data)
        
        # Set RGB to white for all pixels
        result[:, :, 0] = 255  # R
        result[:, :, 1] = 255  # G
        result[:, :, 2] = 255  # B
        
        # CRITICAL: Copy original alpha channel exactly as-is
        result[:, :, 3] = alpha
        
        # Create image from array
        result_img = Image.fromarray(result, mode='RGBA')
        
        # Save as PNG with transparency
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=False)
        output.seek(0)
        
        # Stats
        total_pixels = alpha.size
        transparent_pixels = np.sum(alpha == 0)
        semi_transparent = np.sum((alpha > 0) & (alpha < 255))
        fully_opaque = np.sum(alpha == 255)
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='solid_white.png'
        )
        
        response.headers['X-Total-Pixels'] = str(total_pixels)
        response.headers['X-Transparent-Pixels'] = str(transparent_pixels)
        response.headers['X-Semi-Transparent-Pixels'] = str(semi_transparent)
        response.headers['X-Opaque-Pixels'] = str(fully_opaque)
        response.headers['X-Output-Color'] = 'WHITE'
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/check-transparency', methods=['POST'])
def check_transparency():
    """Check transparency information of uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream)
        
        original_mode = img.mode
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        data = np.array(img)
        alpha = data[:, :, 3]
        
        return jsonify({
            "original_mode": original_mode,
            "converted_mode": "RGBA",
            "dimensions": f"{img.width}x{img.height}",
            "total_pixels": int(alpha.size),
            "transparency_info": {
                "fully_transparent": int(np.sum(alpha == 0)),
                "semi_transparent": int(np.sum((alpha > 0) & (alpha < 255))),
                "fully_opaque": int(np.sum(alpha == 255))
            },
            "has_transparency": bool(np.any(alpha < 255)),
            "min_alpha": int(np.min(alpha)),
            "max_alpha": int(np.max(alpha)),
            "avg_alpha": float(np.mean(alpha))
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/debug-smart-color', methods=['POST'])
def debug_smart_color():
    """Debug version - shows why smart-color-replace fails"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        target_type = request.form.get('target_type', 'dark').lower()
        
        img = Image.open(file.stream).convert('RGBA')
        rgb = img.convert('RGB')
        rgb_array = np.array(rgb)
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Test each condition
        kernel_size = 5
        h_var = cv2.blur(h**2, (kernel_size, kernel_size)) - cv2.blur(h, (kernel_size, kernel_size))**2
        s_var = cv2.blur(s**2, (kernel_size, kernel_size)) - cv2.blur(s, (kernel_size, kernel_size))**2
        v_var = cv2.blur(v**2, (kernel_size, kernel_size)) - cv2.blur(v, (kernel_size, kernel_size))**2
        total_var = h_var + s_var + v_var
        
        solid_mask = total_var < 100
        colored_mask = s > 30
        
        if target_type == 'light':
            brightness_mask = v > 150
        else:
            brightness_mask = v < 120
        
        target_mask = solid_mask & colored_mask & brightness_mask
        
        total_pixels = h.size
        
        return jsonify({
            "image_size": f"{rgb_array.shape[1]}x{rgb_array.shape[0]}",
            "total_pixels": int(total_pixels),
            "results": {
                "solid_pixels": int(np.sum(solid_mask)),
                "solid_percentage": f"{(np.sum(solid_mask)/total_pixels)*100:.2f}%",
                "colored_pixels": int(np.sum(colored_mask)),
                "colored_percentage": f"{(np.sum(colored_mask)/total_pixels)*100:.2f}%",
                "brightness_matched": int(np.sum(brightness_mask)),
                "brightness_percentage": f"{(np.sum(brightness_mask)/total_pixels)*100:.2f}%",
                "final_matched": int(np.sum(target_mask)),
                "final_percentage": f"{(np.sum(target_mask)/total_pixels)*100:.2f}%"
            },
            "diagnosis": "PASS" if np.sum(target_mask) > 0 else "FAIL - No pixels meet all criteria",
            "likely_reason": "Image has gradients (not solid colors)" if np.sum(solid_mask) < (total_pixels * 0.1) else "Check brightness_type parameter"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/replace-dark-to-white', methods=['POST'])
def replace_dark():
    """Replace dark colors with white"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        threshold = int(request.form.get('threshold', DEFAULT_THRESHOLD))
        
        if threshold < 0 or threshold > 255:
            return jsonify({"error": "Threshold must be between 0 and 255"}), 400
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        dark_mask = (r < threshold) & (g < threshold) & (b < threshold)
        
        data[dark_mask, 0:3] = [255, 255, 255]
        
        result_img = Image.fromarray(data, 'RGBA')
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_image.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route('/replace-light-to-dark', methods=['POST'])
def replace_light():
    """Replace light/white colors with dark/black"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        threshold = int(request.form.get('threshold', 200))
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        light_mask = (r > threshold) & (g > threshold) & (b > threshold)
        
        data[light_mask, 0:3] = [0, 0, 0]
        
        result_img = Image.fromarray(data, 'RGBA')
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='light_to_dark.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route('/replace-to-color', methods=['POST'])
def replace_to_color():
    """
    Replace dark or light colors with any target color
    RECOMMENDED: This is the most reliable endpoint
    
    Parameters:
    - image: file (required)
    - target_hue: 0-360 (required) - color to change TO
    - brightness_type: 'dark' or 'light' (default: 'dark')
    - threshold: 0-255 (default: 100 for dark, 200 for light)
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Get parameters
        target_hue = int(request.form.get('target_hue', 0))
        brightness_type = request.form.get('brightness_type', 'dark').lower()
        
        if brightness_type == 'light':
            threshold = int(request.form.get('threshold', 200))
        else:
            threshold = int(request.form.get('threshold', 100))
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        rgb_img = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        
        # Convert to numpy array
        rgb_array = np.array(rgb_img)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Create mask based on brightness
        if brightness_type == 'light':
            # Find light pixels
            brightness_mask = v > threshold
        else:
            # Find dark pixels
            brightness_mask = v < threshold
        
        # Check if any pixels match
        if not np.any(brightness_mask):
            return jsonify({
                "error": f"No {brightness_type} pixels found with threshold {threshold}",
                "suggestion": f"Try adjusting threshold or use brightness_type='{('light' if brightness_type == 'dark' else 'dark')}'"
            }), 400
        
        # Convert target hue to OpenCV range (0-180)
        target_hue_cv = (target_hue % 360) / 2
        
        # Replace hue for matching pixels
        h[brightness_mask] = target_hue_cv
        
        # Boost saturation to make color visible (at least 100)
        s[brightness_mask] = np.maximum(s[brightness_mask], 100)
        
        # Rebuild HSV
        hsv[:,:,0] = h
        hsv[:,:,1] = s
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        # Reapply alpha channel
        if alpha:
            result_img.putalpha(alpha)
        
        # Save and return
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        pixels_changed = np.sum(brightness_mask)
        total_pixels = v.size
        percentage = (pixels_changed / total_pixels) * 100
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'{brightness_type}_to_hue{target_hue}.png'
        )
        
        response.headers['X-Pixels-Changed'] = str(pixels_changed)
        response.headers['X-Change-Percentage'] = f"{percentage:.2f}%"
        response.headers['X-Target-Hue'] = str(target_hue)
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/smart-color-replace', methods=['POST'])
def smart_color_replace():
    """
    Auto-detect dominant solid color and replace it
    Fixed version with better saturation handling
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        new_hue = int(request.form.get('new_hue', 0))
        target_type = request.form.get('target_type', 'dark').lower()
        
        # Optional: allow custom saturation threshold
        min_saturation = int(request.form.get('min_saturation', 10))  # Lowered from 30
        
        img = Image.open(file.stream).convert('RGBA')
        rgb = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        rgb_array = np.array(rgb)
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Calculate variance to find solid colors
        kernel_size = 5
        h_var = cv2.blur(h**2, (kernel_size, kernel_size)) - cv2.blur(h, (kernel_size, kernel_size))**2
        s_var = cv2.blur(s**2, (kernel_size, kernel_size)) - cv2.blur(s, (kernel_size, kernel_size))**2
        v_var = cv2.blur(v**2, (kernel_size, kernel_size)) - cv2.blur(v, (kernel_size, kernel_size))**2
        
        total_var = h_var + s_var + v_var
        
        # Masks
        solid_mask = total_var < 100
        colored_mask = s > min_saturation  # FIXED: Lower threshold
        
        if target_type == 'light':
            brightness_mask = v > 150
        else:
            brightness_mask = v < 120
        
        target_mask = solid_mask & colored_mask & brightness_mask
        
        # If still no match, try without solid requirement
        if not np.any(target_mask):
            target_mask = colored_mask & brightness_mask
            
            if not np.any(target_mask):
                # Last resort: ignore saturation requirement for very desaturated images
                target_mask = brightness_mask
                
                if not np.any(target_mask):
                    return jsonify({
                        "error": f"No {target_type} pixels found",
                        "suggestion": f"Try target_type='{('light' if target_type == 'dark' else 'dark')}'"
                    }), 400
        
        # Find dominant hue
        valid_hues = h[target_mask]
        hist, bin_edges = np.histogram(valid_hues, bins=36, range=(0, 180))
        dominant_bin = np.argmax(hist)
        dominant_hue = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
        
        # Match similar hues (more lenient tolerance)
        hue_diff = np.abs(h - dominant_hue)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        hue_match = hue_diff < 40  # Increased from 20
        
        final_mask = target_mask & hue_match
        
        # If still no match after hue filtering, just use brightness mask
        if not np.any(final_mask):
            final_mask = target_mask
        
        # Replace hue
        hsv[:,:,0][final_mask] = new_hue / 2
        
        # Boost saturation to make color visible
        hsv[:,:,1][final_mask] = np.maximum(hsv[:,:,1][final_mask], 100)
        
        # Convert back
        result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        if alpha:
            result_img.putalpha(alpha)
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        pixels_changed = np.sum(final_mask)
        percentage = (pixels_changed / h.size) * 100
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'smart_replaced_{target_type}.png'
        )
        
        response.headers['X-Detected-Hue'] = str(int(dominant_hue * 2))
        response.headers['X-Pixels-Changed'] = str(int(pixels_changed))
        response.headers['X-Change-Percentage'] = f"{percentage:.2f}%"
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/invert-colors', methods=['POST'])
def invert_colors():
    """Invert all colors (create negative)"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        # Invert RGB, keep alpha
        data[:, :, 0] = 255 - data[:, :, 0]
        data[:, :, 1] = 255 - data[:, :, 1]
        data[:, :, 2] = 255 - data[:, :, 2]
        
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='inverted.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
