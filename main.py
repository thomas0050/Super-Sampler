from flask import *
import os
from PIL import Image
from PIL import ImageCms
from ISR.models import *
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.python.client import device_lib

device_lib.list_local_devices()
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('home.html')


@app.route("/upload", methods=["POST"])
def upload():
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            saved_img = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # ML
            img = Image.open(saved_img)
            lr_img = np.array(img)

            model = RDN(weights='psnr-large')
            sr_img = model.predict(lr_img)
            img1 = Image.fromarray(sr_img)
            img1.save("uploads/save.jpg")
            filename = "save.jpg"
            return redirect(url_for('uploaded_file',
                                    filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(debug=True)
