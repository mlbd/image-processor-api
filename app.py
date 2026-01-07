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
