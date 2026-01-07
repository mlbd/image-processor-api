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
    More complex but may not work with all images
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
        colored_mask = s > 30
        
        if target_type == 'light':
            brightness_mask = v > 150
        else:
            brightness_mask = v < 120
        
        target_mask = solid_mask & colored_mask & brightness_mask
        
        if not np.any(target_mask):
            return jsonify({
                "error": f"No solid {target_type} colored areas found",
                "suggestion": f"Try target_type='{('light' if target_type == 'dark' else 'dark')}' or use /replace-to-color instead",
                "hint": "/replace-to-color is more reliable for most images"
            }), 400
        
        # Find dominant hue
        valid_hues = h[target_mask]
        hist, bin_edges = np.histogram(valid_hues, bins=36, range=(0, 180))
        dominant_bin = np.argmax(hist)
        dominant_hue = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
        
        # Match similar hues
        hue_diff = np.abs(h - dominant_hue)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        hue_match = hue_diff < 20
        
        final_mask = target_mask & hue_match
        
        # Replace hue
        hsv[:,:,0][final_mask] = new_hue / 2
        
        # Convert back
        result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        if alpha:
            result_img.putalpha(alpha)
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'smart_replaced_{target_type}.png'
        )
        
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
