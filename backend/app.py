from flask import Flask
from flask_cors import CORS
from multimodal_routes import multimodal_routes

app = Flask(__name__)
CORS(app)

app.register_blueprint(multimodal_routes)

if __name__ == "__main__":
    app.run(debug=True)
