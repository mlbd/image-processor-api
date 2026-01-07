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
    Detects edges (internal lines + outer strokes) and inverts them
    
    Parameters:
    - image: file (required)
    - base_color: 'black' or 'white' (default: 'black')
    - edge_threshold: 10-255 (default: 30, lower=more sensitive)
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        base_color = request.form.get('base_color', 'black').lower()
        edge_threshold = int(request.form.get('edge_threshold', 30))
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        
        # Get alpha channel and RGB
        rgb = img.convert('RGB')
        alpha = img.split()[3]
        rgb_array = np.array(rgb)
        alpha_array = np.array(alpha)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Canny or Sobel
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
        
        # Also detect edges in alpha channel (outer boundaries)
        alpha_edges = cv2.Canny(alpha_array, 50, 150)
        
        # Combine both edge detections
        all_edges = cv2.bitwise_or(edges, alpha_edges)
        
        # Dilate edges slightly to make them more visible
        kernel = np.ones((2, 2), np.uint8)
        all_edges = cv2.dilate(all_edges, kernel, iterations=1)
        
        # Create result image
        result = np.zeros((img.height, img.width, 4), dtype=np.uint8)
        
        # Set base color
        if base_color == 'white':
            result[:, :, 0:3] = 255  # RGB = white
        else:
            result[:, :, 0:3] = 0    # RGB = black
        
        # Set contrasting color for edges
        edge_mask = all_edges > 0
        if base_color == 'white':
            result[edge_mask, 0:3] = [0, 0, 0]    # Black edges on white
        else:
            result[edge_mask, 0:3] = [255, 255, 255]  # White edges on black
        
        # Apply original alpha channel
        result[:, :, 3] = alpha_array
        
        # Create final image
        result_img = Image.fromarray(result, 'RGBA')
        
        # Save
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'solid_{base_color}_readable.png'
        )
        
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
