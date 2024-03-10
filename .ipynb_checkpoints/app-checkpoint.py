from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("./conveyor_model.h5")

def process_image(file_path):
    img = image.load_img(file_path, target_size=(320, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def make_prediction(img_array):
    prediction = model.predict(img_array)
    return prediction[0][0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        f = request.files["file"]
        file_path = f"./uploads/{secure_filename(f.filename)}"
        f.save(file_path)
        
        img_array = process_image(file_path)
        result = make_prediction(img_array)

        return render_template("result.html", result=result)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
