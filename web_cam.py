from flask import Flask, render_template, Response
import tensorflow 
import cv2

app = Flask(__name__)

model = tensorflow.keras.models.load_model('conveyor_model.h5')
model.make_predict_function()

camera = cv2.VideoCapture('http://192.0.0.4:8080/video')
frame_count = 0  # Counter to keep track of frames

def predict_label(img):
    img = cv2.resize(img, (320, 180))
    img = img.astype('float') / 255.0
    img = img.reshape(1, 320, 180, 3)
    prediction = model.predict(img)
    return prediction[0][0]

def generate_frames():
    global frame_count
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_count += 1

            # Skip frames and only predict every 5th frame
            if frame_count % 5 != 0:
                continue

            prediction = predict_label(frame)
            print(prediction)
            if prediction>0.5:
                print("Defect is detected")
            else:
                print("No defect")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Send both frame and prediction as a response
            yield (b'--frame\r\n'
       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n' +
       b'Content-Type: text/plain\r\n\r\n' + f'<script>updatePrediction("{prediction}")</script>'.encode() + b'\r\n')

@app.route("/")
def main():
    return render_template("web_cam.html")

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
