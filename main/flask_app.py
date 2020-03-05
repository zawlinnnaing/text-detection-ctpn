from flask import request, Flask, jsonify
from werkzeug.utils import secure_filename
import app_config as cfg
import os
import predictor
import add_sys_path
from exceptions.flask_exceptions.APIException import APIException
app = Flask(__name__)


ALLOWED_IMG_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['MAX_CONTENT_LENGTH'] = 12 * 1024*1024


@app.errorhandler(APIException)
def handle_api_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/detect", methods=['POST'])
def detect_text():
    os.makedirs(cfg.SAVE_UPLOADED_IMG_DIR, exist_ok=True)
    if 'image' not in request.files:
        raise APIException("File not uploaded", status_code=402)
    file = request.files['image']
    if file.filename == "":
        raise APIException("Invalid file name", status_code=402)
    if not check_img_ext(file.filename):
        raise APIException("Invalid file type", status_code=402)
    filename = secure_filename(file.filename)
    filename = os.path.join(cfg.SAVE_UPLOADED_IMG_DIR, filename)
    file.save(filename)
    stored_in = predictor.predict(filename)
    os.remove(filename)
    return jsonify({"success": True, "stored_in": stored_in})


def check_img_ext(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMG_EXTENSIONS
