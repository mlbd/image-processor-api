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
        "version": "2.0.0",
        "authentication": "required" if API_KEY else "not required",
        "endpoints": {
            "/health": "GET - Health check",
            "/admin": "GET - Admin panel (for keep-alive)",
            "/replace-dark-to-white": "POST - Replace dark colors with white",
            "/replace-color": "POST - Replace specific color with another",
            "/invert-colors": "POST - Invert all colors (negative)",
            "/make-for-white-bg": "POST - Optimize logo for white background/T-shirt",
            "/make-for-black-bg": "POST - Optimize logo for black background/T-shirt",
            "/auto-adjust": "POST - Smart auto-adjust for target background",
            "/smart-color-replace": "POST - Auto-detect and replace solid colors",
            "/replace-color-pro": "POST - Professional color replacement (preserves quality)"
        },
        "usage": {
            "authentication": "Add 'X-API-Key' header or 'api_key' form field (if enabled)",
            "image_upload": "Use 'image' field in form-data",
            "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Service is running"
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
        "🎪 The pixel circus is open for business!",
        "🧙‍♂️ Abracadabra! Your image processor is alive!",
        "🎯 Bulls-eye! Service is up and pixel-perfect!",
        "🌈 Converting darkness to light since... well, 10 minutes ago!",
        "⚡ Zap! Another successful keep-alive ping!",
        "🎸 Rock on! This service is still jamming!"
    ]
    
    return jsonify({
        "status": "alive_and_kicking",
        "message": random.choice(funny_messages),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service_mood": random.choice(["Excellent", "Spectacular", "Fantastic", "Energetic"]),
        "pixels_processed_today": random.randint(1000, 9999),
        "coffee_level": f"{random.randint(60, 100)}%",
        "motivation": "Over 9000! ☕"
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

@app.route('/replace-color', methods=['POST'])
def replace_specific_color():
    """Replace a specific color with another color"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        tolerance = int(request.form.get('tolerance', 30))
        target = request.form.get('target_color', '0,0,0')
        replacement = request.form.get('replacement_color', '255,255,255')
        
        target_rgb = [int(x) for x in target.split(',')]
        replacement_rgb = [int(x) for x in replacement.split(',')]
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        tr, tg, tb = target_rgb
        distance = np.sqrt((r - tr)**2 + (g - tg)**2 + (b - tb)**2)
        
        mask = distance < tolerance
        data[mask, 0:3] = replacement_rgb
        
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

@app.route('/smart-color-replace', methods=['POST'])
def smart_color_replace():
    """
    Automatically detect and replace the main SOLID color (dark or light)
    - Ignores gradients and multi-color areas
    - Only replaces uniform/solid colored regions
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        # Parameters
        new_hue = int(request.form.get('new_hue', 0))
        target_type = request.form.get('target_type', 'dark').lower()
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        
        rgb = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        rgb_array = np.array(rgb)
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Calculate local variance for solid color detection
        kernel_size = 5
        h_variance = cv2.blur(h**2, (kernel_size, kernel_size)) - cv2.blur(h, (kernel_size, kernel_size))**2
        s_variance = cv2.blur(s**2, (kernel_size, kernel_size)) - cv2.blur(s, (kernel_size, kernel_size))**2
        v_variance = cv2.blur(v**2, (kernel_size, kernel_size)) - cv2.blur(v, (kernel_size, kernel_size))**2
        
        total_variance = h_variance + s_variance + v_variance
        
        # Solid color mask
        variance_threshold = 100
        solid_mask = total_variance < variance_threshold
        
        # Colored pixels only
        saturation_threshold = 30
        colored_mask = s > saturation_threshold
        
        # Dark or light filter
        if target_type == 'light':
            brightness_mask = v > 150
        else:
            brightness_mask = v < 120
        
        # Combine masks
        target_mask = solid_mask & colored_mask & brightness_mask
        
        if not np.any(target_mask):
            return jsonify({
                "error": f"No solid {target_type} colored areas found",
                "suggestion": f"Try 'target_type': '{'light' if target_type == 'dark' else 'dark'}'"
            }), 400
        
        # Find dominant hue
        valid_hues = h[target_mask]
        hist, bin_edges = np.histogram(valid_hues, bins=36, range=(0, 180))
        dominant_hue_bin = np.argmax(hist)
        dominant_hue = (bin_edges[dominant_hue_bin] + bin_edges[dominant_hue_bin + 1]) / 2
        
        # Create final mask
        hue_tolerance = 20
        hue_diff = np.abs(h - dominant_hue)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        hue_match_mask = hue_diff < hue_tolerance
        
        final_mask = target_mask & hue_match_mask
        
        # Replace hue only
        hsv[:,:,0][final_mask] = new_hue / 2
        
        # Convert back
        result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        if alpha:
            result_img.putalpha(alpha)
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        # Stats
        total_pixels = h.size
        replaced_pixels = np.sum(final_mask)
        percentage = (replaced_pixels / total_pixels) * 100
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'smart_replaced_{target_type}.png'
        )
        
        response.headers['X-Detected-Hue'] = str(int(dominant_hue * 2))
        response.headers['X-Pixels-Changed'] = str(replaced_pixels)
        response.headers['X-Change-Percentage'] = f"{percentage:.2f}%"
        
        return response
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
