from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)

# Environment variables
API_KEY = os.environ.get('API_KEY', None)  # Set this in Render dashboard
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE_MB', 10)) * 1024 * 1024
DEFAULT_THRESHOLD = int(os.environ.get('DEFAULT_THRESHOLD', 100))
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def verify_api_key():
    """Verify API key if set in environment"""
    if API_KEY:
        # Check in headers first, then form data
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
            "/auto-adjust": "POST - Smart auto-adjust for target background"
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
        "🎸 Rock on! This service is still jamming!",
        "🍕 Serving hot pixels, not pizza!",
        "🦸‍♂️ Image processing superhero reporting for duty!",
        "🎲 Lucky you! Service is still running!",
        "🏆 Champion status: Still undefeated against sleep mode!",
        "🎉 Party's not over! Service is still dancing!"
    ]
    
    funny_tips = [
        "Pro tip: Black pixels hate becoming white. They call it 'identity theft'.",
        "Did you know? Images processed here are 100% organic and free-range.",
        "Fun fact: We're powered by enthusiasm and Python magic!",
        "Reminder: No pixels were harmed in this keep-alive check.",
        "Secret: The images actually process themselves. We just take credit.",
        "Breaking: Local image processor refuses to sleep, citing 'too much caffeine'.",
        "Insider info: The pixels throw a party every time you ping this endpoint.",
        "Life hack: Always be nice to your image processor. They remember.",
        "Plot twist: The service was awake this whole time!",
        "Exclusive: Service claims it 'doesn't even need sleep anyway'."
    ]
    
    status_emojis = ["✨", "🌟", "⭐", "💫", "🔥", "💪", "🎊", "🎈", "🌺", "🦄"]
    
    return jsonify({
        "status": "alive_and_kicking",
        "message": random.choice(funny_messages),
        "tip": random.choice(funny_tips),
        "emoji": random.choice(status_emojis),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_excuse": "I was just resting my eyes, I swear!",
        "service_mood": random.choice(["Excellent", "Spectacular", "Fantastic", "Phenomenal", "Energetic"]),
        "pixels_processed_today": random.randint(1000, 9999),
        "coffee_level": f"{random.randint(60, 100)}%",
        "motivation_level": "Over 9000!",
        "secret_message": "🤫 You found the secret admin endpoint!",
        "fun_fact": f"This is ping #{random.randint(1, 999)} today!",
        "next_nap_in": "Not today, Satan! ☕"
    })

@app.route('/smart-color-replace', methods=['POST'])
def smart_color_replace():
    """
    Automatically detect and replace the main SOLID color (dark or light)
    - Ignores gradients and multi-color areas
    - Only replaces uniform/solid colored regions
    - Can target either dark colors or light colors
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        # Parameters
        new_hue = int(request.form.get('new_hue', 0))  # Target color to replace with
        target_type = request.form.get('target_type', 'dark').lower()  # 'dark' or 'light'
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        
        rgb = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        rgb_array = np.array(rgb)
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Step 1: Filter for solid colors only (ignore gradients)
        # Check local variance - solid areas have low variance
        kernel_size = 5
        
        # Calculate local variance for each channel
        h_variance = cv2.blur(h**2, (kernel_size, kernel_size)) - cv2.blur(h, (kernel_size, kernel_size))**2
        s_variance = cv2.blur(s**2, (kernel_size, kernel_size)) - cv2.blur(s, (kernel_size, kernel_size))**2
        v_variance = cv2.blur(v**2, (kernel_size, kernel_size)) - cv2.blur(v, (kernel_size, kernel_size))**2
        
        # Total variance - low values = solid color area
        total_variance = h_variance + s_variance + v_variance
        
        # Solid color mask (low variance areas)
        variance_threshold = 100  # Adjust this for strictness (lower = more strict)
        solid_mask = total_variance < variance_threshold
        
        # Step 2: Filter by saturation (colored pixels only, not grays)
        saturation_threshold = 30
        colored_mask = s > saturation_threshold
        
        # Step 3: Filter by brightness (dark or light)
        if target_type == 'light':
            # Light/whitish colors (high brightness)
            brightness_mask = v > 150  # Bright pixels
        else:
            # Dark colors (low brightness)
            brightness_mask = v < 120  # Dark pixels
        
        # Combine all masks
        target_mask = solid_mask & colored_mask & brightness_mask
        
        # Check if we found any valid pixels
        if not np.any(target_mask):
            return jsonify({
                "error": f"No solid {target_type} colored areas found in the image",
                "suggestion": f"Try changing 'target_type' to '{'light' if target_type == 'dark' else 'dark'}'"
            }), 400
        
        # Step 4: Find the dominant hue among valid pixels
        valid_hues = h[target_mask]
        
        # Use histogram to find most common hue
        hist, bin_edges = np.histogram(valid_hues, bins=36, range=(0, 180))
        dominant_hue_bin = np.argmax(hist)
        dominant_hue = (bin_edges[dominant_hue_bin] + bin_edges[dominant_hue_bin + 1]) / 2
        
        # Step 5: Create final mask for pixels similar to dominant hue
        hue_tolerance = 20  # How similar colors should be to get replaced
        
        # Find pixels matching dominant hue
        hue_diff = np.abs(h - dominant_hue)
        # Handle hue wrap-around (0° and 180° are close)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        
        hue_match_mask = hue_diff < hue_tolerance
        
        # Final mask: solid + colored + right brightness + matching hue
        final_mask = target_mask & hue_match_mask
        
        # Step 6: Replace hue ONLY (preserve saturation and value for natural look)
        hsv[:,:,0][final_mask] = new_hue / 2  # Convert to OpenCV range (0-180)
        
        # Step 7: Convert back to RGB
        result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        # Reapply alpha channel
        if alpha:
            result_img.putalpha(alpha)
        
        # Return result with metadata
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        # Calculate some stats for debugging
        total_pixels = h.size
        replaced_pixels = np.sum(final_mask)
        percentage = (replaced_pixels / total_pixels) * 100
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'smart_replaced_{target_type}.png'
        )
        
        # Add metadata headers
        response.headers['X-Detected-Hue'] = str(int(dominant_hue * 2))
        response.headers['X-Pixels-Changed'] = str(replaced_pixels)
        response.headers['X-Change-Percentage'] = f"{percentage:.2f}%"
        
        return response
        
    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500
        
@app.route('/replace-dark-to-white', methods=['POST'])
def replace_dark():
    """Replace dark colors with white (for making logos visible on dark backgrounds)"""
    # Verify API key
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        threshold = int(request.form.get('threshold', DEFAULT_THRESHOLD))
        
        if threshold < 0 or threshold > 255:
            return jsonify({"error": "Threshold must be between 0 and 255"}), 400
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided. Send as 'image' in form-data"}), 400
        
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
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

@app.route('/replace-color-pro', methods=['POST'])
def replace_color_professional():
    """Professional color replacement - preserves quality, gradients, shadows"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        # Get parameters
        target_hue = int(request.form.get('target_hue', 210))  # Blue default
        new_hue = int(request.form.get('new_hue', 0))          # Red default
        hue_tolerance = int(request.form.get('hue_tolerance', 30))
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        
        # Separate alpha channel
        rgb = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        
        # Convert to numpy array
        rgb_array = np.array(rgb)
        
        # Convert RGB to HSV (this is the magic!)
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Extract H, S, V channels
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Create mask for target color
        # Handle hue wrap-around (hue is 0-180 in OpenCV)
        target_hue_cv = target_hue / 2  # Convert to OpenCV range
        tolerance_cv = hue_tolerance / 2
        
        # Find pixels within hue range
        lower_bound = target_hue_cv - tolerance_cv
        upper_bound = target_hue_cv + tolerance_cv
        
        if lower_bound < 0:
            mask = ((h >= 180 + lower_bound) | (h <= upper_bound))
        elif upper_bound > 180:
            mask = ((h >= lower_bound) | (h <= upper_bound - 180))
        else:
            mask = ((h >= lower_bound) & (h <= upper_bound))
        
        # Also consider saturation (ignore very desaturated/gray pixels)
        mask = mask & (s > 30)  # Only change colored pixels, not grays
        
        # Replace hue ONLY (keep saturation and value!)
        h[mask] = new_hue / 2  # Convert to OpenCV range
        
        # Reconstruct HSV
        hsv[:,:,0] = h
        # s and v remain unchanged!
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Convert to PIL Image
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        # Reapply alpha channel if existed
        if alpha:
            result_img.putalpha(alpha)
        
        # Save and return
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='color_replaced_pro.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500
        
@app.route('/replace-color', methods=['POST'])
def replace_specific_color():
    """Replace a specific color with another color"""
    # Verify API key
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
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

@app.route('/invert-colors', methods=['POST'])
def invert_colors():
    """Invert all colors (create negative/opposite colors)"""
    # Verify API key
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        # Invert RGB channels (keep alpha)
        data[:, :, 0] = 255 - data[:, :, 0]  # Red
        data[:, :, 1] = 255 - data[:, :, 1]  # Green
        data[:, :, 2] = 255 - data[:, :, 2]  # Blue
        # Alpha channel stays the same
        
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='inverted_logo.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route('/make-for-white-bg', methods=['POST'])
def make_for_white_background():
    """Convert logo to work on white background/T-shirt (make light colors dark)"""
    # Verify API key
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        # Allow custom threshold
        light_threshold = int(request.form.get('light_threshold', 200))
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        
        # Calculate brightness (0-255)
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        
        # Find light pixels
        light_mask = brightness > light_threshold
        
        # Make light pixels dark (invert them)
        data[light_mask, 0] = 255 - data[light_mask, 0]
        data[light_mask, 1] = 255 - data[light_mask, 1]
        data[light_mask, 2] = 255 - data[light_mask, 2]
        
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='logo_for_white_bg.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route('/make-for-black-bg', methods=['POST'])
def make_for_black_background():
    """Convert logo to work on black background/T-shirt (make dark colors light)"""
    # Verify API key
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        # Allow custom threshold
        dark_threshold = int(request.form.get('dark_threshold', 100))
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        
        # Calculate brightness
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        
        # Find dark pixels
        dark_mask = brightness < dark_threshold
        
        # Make dark pixels light (invert them)
        data[dark_mask, 0] = 255 - data[dark_mask, 0]
        data[dark_mask, 1] = 255 - data[dark_mask, 1]
        data[dark_mask, 2] = 255 - data[dark_mask, 2]
        
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='logo_for_black_bg.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route('/auto-adjust', methods=['POST'])
def auto_adjust():
    """Smart auto-detection: Optimizes logo for target background (white or black)"""
    # Verify API key
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img).copy()
        
        # Analyze the logo
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        
        # Calculate average brightness of non-transparent pixels
        non_transparent = a > 0
        if not np.any(non_transparent):
            return jsonify({"error": "Image is completely transparent"}), 400
        
        avg_brightness = np.mean(brightness[non_transparent])
        
        # Determine if logo is mostly light or dark
        is_light_logo = avg_brightness > 127
        
        # Get target background (default: white)
        target_bg = request.form.get('target_background', 'white').lower()
        
        # Create appropriate version
        if target_bg == 'black':
            # For black background, ensure logo is light/visible
            if not is_light_logo:
                # Logo is dark, so brighten it
                dark_mask = brightness < 100
                data[dark_mask, 0:3] = 255 - data[dark_mask, 0:3]
            filename = 'logo_for_black_tshirt.png'
        else:
            # For white background, ensure logo is dark/visible
            if is_light_logo:
                # Logo is light, so darken it
                light_mask = brightness > 150
                data[light_mask, 0:3] = 255 - data[light_mask, 0:3]
            filename = 'logo_for_white_tshirt.png'
        
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
