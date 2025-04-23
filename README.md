# Pneumonia Detection Flask App

This project is a Flask-based web application for detecting pneumonia from chest X-ray images using a pre-trained Convolutional Neural Network (CNN) model.

## Features
- Upload chest X-ray images via a web interface.
- Predict whether the image indicates "PNEUMONIA" or "NORMAL".
- Display the uploaded image and prediction result on the web page.

## Project Structure
```
.
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Untitled0.ipynb        # Jupyter Notebook for model training
├── pneumonia_model/
│   └── pneumonia_cnn_model.h5  # Pre-trained CNN model
├── static/
│   ├── styles.css         # CSS for styling
│   └── uploaded/          # Directory for uploaded images
├── templates/
│   └── index.html         # HTML template for the web interface
├── test/
│   ├── nomal/             # Test images classified as normal
│   └── pneumonia/         # Test images classified as pneumonia
└── uploads/               # Directory for user-uploaded images
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd pneumonia_flask_app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the pre-trained model (`pneumonia_cnn_model.h5`) is in the `pneumonia_model/` directory.

## Usage
1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000/`.

3. Upload a chest X-ray image to get the prediction result.

## Model Training
- The Jupyter Notebook (`Untitled0.ipynb`) contains the code for training the CNN model using TensorFlow/Keras.
- The model is trained on chest X-ray datasets and saved as `pneumonia_cnn_model.h5`.

## License
This project is licensed under the MIT License.