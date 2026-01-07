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
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        # Check for image first
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "received_files": list(request.files.keys()),
                "received_form": list(request.form.keys())
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Get parameters with better validation
        try:
            new_hue = int(request.form.get('new_hue', 0))
        except (ValueError, TypeError):
            return jsonify({"error": "new_hue must be an integer (0-360)"}), 400
        
        target_type = request.form.get('target_type', 'dark').lower()
        
        if target_type not in ['dark', 'light']:
            return jsonify({"error": "target_type must be 'dark' or 'light'"}), 400
        
        # Open and validate image
        try:
            img = Image.open(file.stream).convert('RGBA')
        except Exception as e:
            return jsonify({"error": "Invalid image file", "details": str(e)}), 400
        
        # Get RGB and alpha
        rgb = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        rgb_array = np.array(rgb)
        
        # Check if image is not empty
        if rgb_array.size == 0:
            return jsonify({"error": "Image is empty"}), 400
        
        # Convert to HSV
        try:
            hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        except Exception as e:
            return jsonify({"error": "Failed to convert image to HSV", "details": str(e)}), 500
        
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Calculate local variance for solid color detection
        kernel_size = 5
        try:
            h_blur = cv2.blur(h, (kernel_size, kernel_size))
            h_blur_sq = cv2.blur(h**2, (kernel_size, kernel_size))
            h_variance = h_blur_sq - h_blur**2
            
            s_blur = cv2.blur(s, (kernel_size, kernel_size))
            s_blur_sq = cv2.blur(s**2, (kernel_size, kernel_size))
            s_variance = s_blur_sq - s_blur**2
            
            v_blur = cv2.blur(v, (kernel_size, kernel_size))
            v_blur_sq = cv2.blur(v**2, (kernel_size, kernel_size))
            v_variance = v_blur_sq - v_blur**2
            
            total_variance = h_variance + s_variance + v_variance
        except Exception as e:
            return jsonify({"error": "Failed to calculate variance", "details": str(e)}), 500
        
        # Solid color mask (low variance = solid)
        variance_threshold = 200
        solid_mask = total_variance < variance_threshold
        
        # Colored pixels only (not gray)
        saturation_threshold = 30
        colored_mask = s > saturation_threshold
        
        # Filter by brightness (dark or light)
        if target_type == 'light':
            brightness_mask = v > 150
        else:
            brightness_mask = v < 120
        
        # Combine all masks
        target_mask = solid_mask & colored_mask & brightness_mask
        
        # Count how many pixels match
        matching_pixels = np.sum(target_mask)
        total_pixels = h.size
        
        if matching_pixels == 0:
            return jsonify({
                "error": f"No solid {target_type} colored areas found in image",
                "suggestion": f"Try target_type='{('light' if target_type == 'dark' else 'dark')}'",
                "debug_info": {
                    "total_pixels": int(total_pixels),
                    "solid_pixels": int(np.sum(solid_mask)),
                    "colored_pixels": int(np.sum(colored_mask)),
                    "brightness_matched": int(np.sum(brightness_mask)),
                    "final_matched": int(matching_pixels)
                }
            }), 400
        
        # Find dominant hue among valid pixels
        valid_hues = h[target_mask]
        
        try:
            hist, bin_edges = np.histogram(valid_hues, bins=36, range=(0, 180))
            dominant_hue_bin = np.argmax(hist)
            dominant_hue = (bin_edges[dominant_hue_bin] + bin_edges[dominant_hue_bin + 1]) / 2
        except Exception as e:
            return jsonify({"error": "Failed to find dominant color", "details": str(e)}), 500
        
        # Create hue matching mask
        hue_tolerance = 20
        hue_diff = np.abs(h - dominant_hue)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        hue_match_mask = hue_diff < hue_tolerance
        
        # Final mask
        final_mask = target_mask & hue_match_mask
        
        replaced_pixels = np.sum(final_mask)
        
        if replaced_pixels == 0:
            return jsonify({
                "error": "No pixels matched the dominant color pattern",
                "debug_info": {
                    "dominant_hue_detected": int(dominant_hue * 2),
                    "target_type": target_type,
                    "pixels_in_target_range": int(matching_pixels)
                }
            }), 400
        
        # Replace hue only (preserve saturation and value)
        hsv[:,:,0][final_mask] = new_hue / 2
        
        # Convert back to RGB
        try:
            result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        except Exception as e:
            return jsonify({"error": "Failed to convert back to RGB", "details": str(e)}), 500
        
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        # Reapply alpha
        if alpha:
            result_img.putalpha(alpha)
        
        # Save to output
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        # Calculate stats
        percentage = (replaced_pixels / total_pixels) * 100
        
        # Create response
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'smart_replaced_{target_type}.png'
        )
        
        # Add metadata headers
        response.headers['X-Detected-Hue'] = str(int(dominant_hue * 2))
        response.headers['X-Pixels-Changed'] = str(int(replaced_pixels))
        response.headers['X-Change-Percentage'] = f"{percentage:.2f}%"
        response.headers['X-Target-Type'] = target_type
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/replace-dark-with-color', methods=['POST'])
def replace_dark_with_color():
    """
    Simple: Replace dark colors with any color (HSV-based for quality)
    Works exactly like replace-dark-to-white but with any target color
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        threshold = int(request.form.get('threshold', 100))
        new_hue = int(request.form.get('new_hue', 0))  # Target color
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        
        rgb = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        rgb_array = np.array(rgb)
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Find dark pixels (simple brightness check)
        dark_mask = v < threshold
        
        if not np.any(dark_mask):
            return jsonify({"error": "No dark pixels found"}), 400
        
        # Replace hue of dark pixels
        hsv[:,:,0][dark_mask] = new_hue / 2
        
        # Boost saturation so color is visible
        hsv[:,:,1][dark_mask] = np.maximum(hsv[:,:,1][dark_mask], 100)
        
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
            download_name='color_replaced.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route('/test-smart-color-replace', methods=['POST'])
def test_smart_color_replace():
    """Debug version - shows what's being received"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    return jsonify({
        "received_files": list(request.files.keys()),
        "received_form": dict(request.form),
        "received_headers": dict(request.headers),
        "file_info": {
            "filename": request.files.get('image').filename if 'image' in request.files else None,
            "content_type": request.files.get('image').content_type if 'image' in request.files else None
        } if 'image' in request.files else "No image"
    })
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
