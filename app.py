from flask import Flask, render_template, request
import tensorflow 



app = Flask(__name__)

model = tensorflow.keras.models.load_model('conveyor_model.h5')
model.make_predict_function()

def predict_label(img_path):
    i = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=(320, 180))
    i = tensorflow.keras.preprocessing.image.img_to_array(i) / 255.0
    i = i.reshape(1, 320, 180, 3)
    p = model.predict(i)
    return p

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files.get('my_image')

        if img is None or img.filename == '':
            return render_template("index.html", error="Please upload an image.")

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

        if p[0][0] > 0.5:
            flag = "Defect detected"
        else:
            flag = "No defect"

        return render_template("index.html", prediction=p[0][0], img_path=img_path, flag_variable=flag)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)