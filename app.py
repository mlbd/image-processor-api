from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "Image Color Replacer API",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/replace-dark-to-white": "POST - Replace dark colors with white"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "Service is running"})

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
    try:
        # Get parameters (with defaults)
        threshold = int(request.form.get('threshold', 100))
        
        # Validate threshold
        if threshold < 0 or threshold > 255:
            return jsonify({"error": "Threshold must be between 0 and 255"}), 400
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided. Send as 'image' in form-data"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Process image
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        # Find dark pixels (R, G, B all below threshold)
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        dark_mask = (r < threshold) & (g < threshold) & (b < threshold)
        
        # Replace dark pixels with white (preserve alpha)
        data[dark_mask, 0:3] = [255, 255, 255]
        
        # Convert back to image
        result_img = Image.fromarray(data, 'RGBA')
        
        # Save to bytes
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
    try:
        threshold = int(request.form.get('tolerance', 30))
        
        # Get target and replacement colors (format: "R,G,B")
        target = request.form.get('target_color', '0,0,0')  # Default: black
        replacement = request.form.get('replacement_color', '255,255,255')  # Default: white
        
        target_rgb = [int(x) for x in target.split(',')]
        replacement_rgb = [int(x) for x in replacement.split(',')]
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        # Calculate color distance
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        tr, tg, tb = target_rgb
        distance = np.sqrt((r - tr)**2 + (g - tg)**2 + (b - tb)**2)
        
        # Create mask for similar colors
        mask = distance < threshold
        
        # Replace colors
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
