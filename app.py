import os
import logging
import datetime
import json
import numpy as np
from flask import Flask, render_template, jsonify
from flask_bootstrap import Bootstrap
from models import db

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.json_encoder = NumpyEncoder  # Use the custom JSON encoder

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 300,
    'pool_pre_ping': True,
}

# Initialize extensions
bootstrap = Bootstrap(app)
db.init_app(app)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# Add datetime to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 35 * 1024 * 1024  # 35MB max upload size

# Import routes after app is created to avoid circular imports
from routes import document_routes, query_routes

# Register blueprints
app.register_blueprint(document_routes.bp)
app.register_blueprint(query_routes.bp)

# Define error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle request entity too large error."""
    return render_template('error.html', error="File too large, maximum size is 35MB"), 413

@app.errorhandler(404)
def not_found(e):
    """Handle page not found error."""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle internal server error."""
    app.logger.error(f"500 error: {str(e)}")
    
    # Get more detailed error information
    import traceback
    error_traceback = traceback.format_exc()
    app.logger.error(f"Traceback: {error_traceback}")
    
    # In development mode, show the traceback to assist debugging
    if app.debug:
        error_message = f"Internal server error: {str(e)}<br><pre>{error_traceback}</pre>"
    else:
        error_message = "An unexpected error occurred. Please try again later."
    
    return render_template('error.html', error=error_message), 500

# Root route
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

# If this file is run directly, start the development server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)