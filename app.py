import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

# --- Khởi tạo app Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load model ---
model = load_model('pneumonia_model/pneumonia_cnn_model.h5')
target_size = (150, 150)  # chỉnh theo input model của bạn

# --- Route chính ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Xử lý ảnh
            img = image.load_img(filepath, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Dự đoán
            result = model.predict(img_array)[0][0]
            prediction = "PNEUMONIA" if result >= 0.5 else "NORMAL"

            # Đường dẫn ảnh để hiển thị
            image_url = url_for('static', filename='uploaded/' + filename)

            # Di chuyển ảnh vào static để hiển thị (nếu muốn dùng static)
            static_upload_path = os.path.join('static', 'uploaded')
            os.makedirs(static_upload_path, exist_ok=True)
            os.replace(filepath, os.path.join(static_upload_path, filename))

            image_url = url_for('static', filename=f'uploaded/{filename}')

    return render_template('index.html', prediction=prediction, image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
