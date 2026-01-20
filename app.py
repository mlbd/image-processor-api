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
        "version": "3.0.0",
        "authentication": "required" if API_KEY else "not required",
        "endpoints": {
            "/health": "GET - Health check",
            "/smart-print-ready": "POST - 🎯 SMART print-ready conversion (RECOMMENDED)",
            "/smart-print-ready/analyze": "POST - Analyze logo before processing",
            "/force-solid-black": "POST - Force entire logo to solid black",
            "/force-solid-white": "POST - Force entire logo to solid white",
            "/replace-dark-to-white": "POST - Replace dark colors with white",
            "/replace-light-to-dark": "POST - Replace light colors with dark",
            "/invert-colors": "POST - Invert all colors",
            "/check-transparency": "POST - Check image transparency"
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
    🎯 ULTIMATE PRINT-READY CONVERSION v3
    
    Converts ALL colors to shades of the target print_color.
    - Fill = 100% print_color
    - Borders = graduated shades (90%, 80%, 70%... of print_color)
    
    NO opposite colors remain (no black when white, no white when black)
    
    Parameters:
    - image: file (required)
    - print_color: 'white' or 'black' (required)
    - layers: 2-5 (default: 'auto' - detects from image)
    - contrast_step: 10-30 (default: 15) - percentage difference between layers
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        print_color = request.form.get('print_color', '').lower()
        num_layers = request.form.get('layers', 'auto').lower()
        contrast_step = int(request.form.get('contrast_step', 15))
        
        if print_color not in ['white', 'black']:
            return jsonify({
                "error": "print_color is required",
                "usage": "print_color=white (for dark shirts) or print_color=black (for light shirts)"
            }), 400
        
        # Clamp contrast_step
        contrast_step = max(10, min(30, contrast_step))
        
        # Load image
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        height, width = data.shape[:2]
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        original_alpha = a.copy()
        
        # ============================================================
        # STEP 1: DETECT AND REMOVE BACKGROUND
        # ============================================================
        
        corner_positions = [
            (0, 0), (0, width-1), (height-1, 0), (height-1, width-1),
            (0, width//4), (0, width*3//4),
            (height-1, width//4), (height-1, width*3//4),
            (height//4, 0), (height*3//4, 0),
            (height//4, width-1), (height*3//4, width-1)
        ]
        
        corner_colors = []
        corner_alphas = []
        for y, x in corner_positions:
            if 0 <= y < height and 0 <= x < width:
                corner_colors.append(data[y, x, :3])
                corner_alphas.append(data[y, x, 3])
        
        avg_corner = np.mean(corner_colors, axis=0)
        avg_alpha = np.mean(corner_alphas)
        
        # Detect background type
        if avg_alpha < 128:
            bg_type = "transparent"
            bg_mask = original_alpha < 128
        elif np.mean(avg_corner) > 235:
            bg_type = "white"
            color_diff = np.sqrt(
                (r.astype(np.float32) - 255)**2 +
                (g.astype(np.float32) - 255)**2 +
                (b.astype(np.float32) - 255)**2
            )
            bg_mask = color_diff < 40
        elif np.mean(avg_corner) < 25:
            bg_type = "black"
            color_diff = np.sqrt(
                r.astype(np.float32)**2 +
                g.astype(np.float32)**2 +
                b.astype(np.float32)**2
            )
            bg_mask = color_diff < 40
        else:
            bg_type = "colored"
            color_diff = np.sqrt(
                (r.astype(np.float32) - avg_corner[0])**2 +
                (g.astype(np.float32) - avg_corner[1])**2 +
                (b.astype(np.float32) - avg_corner[2])**2
            )
            bg_mask = color_diff < 50
        
        # Include already transparent pixels
        bg_mask = bg_mask | (original_alpha < 128)
        
        # Clean up background mask with morphology
        kernel = np.ones((5, 5), np.uint8)
        bg_mask_uint8 = bg_mask.astype(np.uint8) * 255
        bg_mask_uint8 = cv2.morphologyEx(bg_mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Flood fill from corners to ensure connected background
        flood_mask = np.zeros((height + 2, width + 2), np.uint8)
        bg_flood = bg_mask_uint8.copy()
        
        # Flood from corners
        for y, x in [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)]:
            if bg_mask_uint8[y, x] > 127:
                cv2.floodFill(bg_flood, flood_mask, (x, y), 255)
        
        bg_mask = bg_flood > 127
        
        logo_mask = ~bg_mask & (original_alpha > 0)
        
        if not np.any(logo_mask):
            return jsonify({"error": "No logo content found after background removal"}), 400
        
        # ============================================================
        # STEP 2: CALCULATE LUMINANCE FOR ALL LOGO PIXELS
        # ============================================================
        
        luminance = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        
        # Get luminance values for logo pixels only
        logo_luminance = luminance[logo_mask]
        
        # ============================================================
        # STEP 3: DETERMINE NUMBER OF LAYERS
        # ============================================================
        
        lum_min = np.min(logo_luminance)
        lum_max = np.max(logo_luminance)
        lum_range = lum_max - lum_min
        lum_std = np.std(logo_luminance)
        
        if num_layers == 'auto':
            # Analyze histogram to find natural clusters
            hist, bins = np.histogram(logo_luminance, bins=32)
            
            # Find significant peaks
            threshold = np.max(hist) * 0.1
            peaks = 0
            in_peak = False
            for count in hist:
                if count > threshold and not in_peak:
                    peaks += 1
                    in_peak = True
                elif count <= threshold:
                    in_peak = False
            
            optimal_layers = max(2, min(5, peaks))
            
            # Also consider luminance range
            if lum_range < 50:
                optimal_layers = 2
            elif lum_range < 100:
                optimal_layers = min(optimal_layers, 3)
        else:
            try:
                optimal_layers = int(num_layers)
                optimal_layers = max(2, min(5, optimal_layers))
            except:
                optimal_layers = 3
        
        # ============================================================
        # STEP 4: K-MEANS CLUSTERING ON LUMINANCE
        # ============================================================
        
        logo_lum_flat = logo_luminance.reshape(-1, 1).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            logo_lum_flat,
            optimal_layers,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS
        )
        
        # Sort centers by luminance
        center_values = centers.flatten()
        center_order = np.argsort(center_values)  # Darkest to lightest
        
        # Create mapping: original label -> sorted rank (0=darkest, n=lightest)
        label_to_rank = {old_label: rank for rank, old_label in enumerate(center_order)}
        
        # ============================================================
        # STEP 5: GENERATE OUTPUT COLORS
        # ============================================================
        
        # Generate graduated shades based on print_color
        output_colors = []
        
        if print_color == 'white':
            # All shades of white (lightest = fill, darker = borders)
            # Layer 0 (darkest original) → lightest shade (most contrast from fill)
            # Layer n (lightest original) → 100% white (fill)
            
            for i in range(optimal_layers):
                # Reverse: darkest original becomes the "outer" shade
                shade_index = optimal_layers - 1 - i
                darkness = shade_index * contrast_step  # 0%, 15%, 30%...
                gray_value = int(255 - (255 * darkness / 100))
                gray_value = max(180, gray_value)  # Never darker than rgb(180,180,180)
                output_colors.append([gray_value, gray_value, gray_value])
            
            # Ensure the lightest (fill) is pure white
            output_colors[-1] = [255, 255, 255]
            
        else:  # print_color == 'black'
            # All shades of black (darkest = fill, lighter = borders)
            # Layer 0 (darkest original) → 100% black (fill)
            # Layer n (lightest original) → lightest shade
            
            for i in range(optimal_layers):
                lightness = i * contrast_step  # 0%, 15%, 30%...
                gray_value = int(255 * lightness / 100)
                gray_value = min(75, gray_value)  # Never lighter than rgb(75,75,75)
                output_colors.append([gray_value, gray_value, gray_value])
            
            # Ensure the darkest (fill) is pure black
            output_colors[0] = [0, 0, 0]
        
        # ============================================================
        # STEP 6: BUILD OUTPUT IMAGE
        # ============================================================
        
        result = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Background = fully transparent
        result[bg_mask] = [0, 0, 0, 0]
        
        # Map each logo pixel to its output color
        logo_indices = np.where(logo_mask)
        logo_y = logo_indices[0]
        logo_x = logo_indices[1]
        
        for i, (y, x) in enumerate(zip(logo_y, logo_x)):
            original_label = labels[i][0]
            rank = label_to_rank[original_label]  # 0=darkest, n-1=lightest
            
            color = output_colors[rank]
            result[y, x] = [color[0], color[1], color[2], 255]
        
        # ============================================================
        # STEP 7: PRESERVE ANTI-ALIASING ON EDGES
        # ============================================================
        
        # Find edge pixels (where alpha was partial in original or at logo boundary)
        logo_mask_uint8 = logo_mask.astype(np.uint8) * 255
        edges = cv2.Canny(logo_mask_uint8, 100, 200)
        edge_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edge_mask = edge_dilated > 0
        
        # Apply original partial transparency to edges for smoother look
        edge_pixels = edge_mask & logo_mask
        # Keep full opacity but could soften if needed:
        # result[edge_pixels, 3] = np.minimum(result[edge_pixels, 3], original_alpha[edge_pixels])
        
        # ============================================================
        # STEP 8: OUTPUT
        # ============================================================
        
        result_img = Image.fromarray(result, 'RGBA')
        
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=True)
        output.seek(0)
        
        # Calculate stats for each layer
        layer_stats = {}
        for rank in range(optimal_layers):
            count = sum(1 for i in range(len(labels)) if label_to_rank[labels[i][0]] == rank)
            layer_stats[f"layer_{rank}"] = {
                "pixels": count,
                "output_color": output_colors[rank],
                "role": "fill" if (print_color == 'white' and rank == optimal_layers-1) or 
                                 (print_color == 'black' and rank == 0) else "border/detail"
            }
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'print_ready_{print_color}.png'
        )
        
        response.headers['X-Print-Color'] = print_color.upper()
        response.headers['X-Background-Type'] = bg_type
        response.headers['X-Layers-Used'] = str(optimal_layers)
        response.headers['X-Contrast-Step'] = f"{contrast_step}%"
        response.headers['X-Luminance-Range'] = f"{lum_min:.0f}-{lum_max:.0f}"
        response.headers['X-Output-Colors'] = str(output_colors)
        response.headers['X-Logo-Pixels'] = str(len(logo_y))
        response.headers['X-BG-Pixels-Removed'] = str(int(np.sum(bg_mask)))
        
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
    Analyze logo - shows detected layers and expected output colors
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        contrast_step = int(request.form.get('contrast_step', 15))
        
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        height, width = data.shape[:2]
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        
        # Simple background detection
        corners = [data[0,0], data[0,-1], data[-1,0], data[-1,-1]]
        avg_corner = np.mean([c[:3] for c in corners], axis=0)
        avg_alpha = np.mean([c[3] for c in corners])
        
        if avg_alpha < 128:
            bg_type = "transparent"
        elif np.mean(avg_corner) > 235:
            bg_type = "white"
        elif np.mean(avg_corner) < 25:
            bg_type = "black"
        else:
            bg_type = "colored"
        
        # Analyze luminance
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        logo_mask = a > 128
        
        if not np.any(logo_mask):
            return jsonify({"error": "No logo content found"}), 400
        
        logo_lum = luminance[logo_mask]
        lum_min, lum_max = float(np.min(logo_lum)), float(np.max(logo_lum))
        lum_std = float(np.std(logo_lum))
        lum_range = lum_max - lum_min
        
        # Estimate layers
        hist, bins = np.histogram(logo_lum, bins=32)
        threshold = np.max(hist) * 0.1
        peaks = 0
        in_peak = False
        for count in hist:
            if count > threshold and not in_peak:
                peaks += 1
                in_peak = True
            elif count <= threshold:
                in_peak = False
        
        optimal_layers = max(2, min(5, peaks))
        if lum_range < 50:
            optimal_layers = 2
        
        # Generate expected output colors
        white_colors = []
        black_colors = []
        
        for i in range(optimal_layers):
            # White print colors
            shade_index = optimal_layers - 1 - i
            darkness = shade_index * contrast_step
            gray = max(180, int(255 - (255 * darkness / 100)))
            white_colors.append(f"rgb({gray},{gray},{gray})")
            
            # Black print colors
            lightness = i * contrast_step
            gray = min(75, int(255 * lightness / 100))
            black_colors.append(f"rgb({gray},{gray},{gray})")
        
        white_colors[-1] = "rgb(255,255,255) [FILL]"
        black_colors[0] = "rgb(0,0,0) [FILL]"
        
        return jsonify({
            "image": {
                "dimensions": f"{width}x{height}",
                "total_pixels": width * height
            },
            "background": {
                "type": bg_type,
                "will_be_removed": True
            },
            "luminance_analysis": {
                "min": lum_min,
                "max": lum_max,
                "range": lum_range,
                "std_dev": lum_std,
                "histogram_peaks": peaks
            },
            "layer_detection": {
                "recommended_layers": optimal_layers,
                "reason": f"Found {peaks} distinct brightness levels, range={lum_range:.0f}"
            },
            "expected_output": {
                "white_print": {
                    "description": "All shades of white/light gray, NO black",
                    "colors_darkest_to_lightest": white_colors,
                    "fill_color": "rgb(255,255,255)",
                    "border_shades": f"Graduated by {contrast_step}% steps"
                },
                "black_print": {
                    "description": "All shades of black/dark gray, NO white",
                    "colors_darkest_to_lightest": black_colors,
                    "fill_color": "rgb(0,0,0)",
                    "border_shades": f"Graduated by {contrast_step}% steps"
                }
            },
            "api_usage": {
                "basic": "POST /smart-print-ready with print_color=white or black",
                "custom_layers": f"Add layers={optimal_layers} to override auto-detection",
                "custom_contrast": "Add contrast_step=10|15|20|25|30 to adjust shade difference"
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
