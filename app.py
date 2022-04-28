from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder='templates')
app.debug = True

ocular_classes = {0: "Normal", 1:"Cataract"}
brain_classes = {0:'glioma_tumor',1:'no_tumor',2:'meningioma_tumor',3:'pituitary_tumor'}

ocular_model = load_model("models//custom_v1.h5")
brain_model = load_model("models//eff3.h5")

def predict_ocular_class(img_path):
    
    i = cv2.imread(img_path)
    im = cv2.resize(i, (224, 224))
    im = np.expand_dims(im, axis=0)
    pred = ocular_model.predict(im)
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    pred_class = ocular_classes[pred[0][0]]
    return pred_class

def predict_brain_class(img_path):
    
    i = cv2.imread(img_path)
    im = cv2.resize(i, (150, 150))
    im = np.expand_dims(im, axis=0)
    pred = brain_model.predict(im)
    print(pred)
    pred = np.argmax(pred,axis=1)
    print(pred)
    pred_class = brain_classes[pred[0]]
    pred_list = pred_class.split("_")
    pred_new = [i.capitalize() for i in pred_list]
    pred_text = " ".join(pred_new)
    return pred_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route("/eye", methods=["POST"])
def eye_upload_file():
    if request.method == 'POST':  
        img = request.files['file']
        img_path = "static/images/" + img.filename
        img.save(img_path)
        
        p = predict_ocular_class(img_path)
        p = p + " Eye"
        return render_template("success.html", prediction=p, filename=img.filename)

@app.route("/brain", methods=["POST"])
def brain_upload_file():
    if request.method == 'POST':  
        img = request.files['file']
        img_path = "static/images/" + img.filename
        img.save(img_path)
        
        p = predict_brain_class(img_path)
        return render_template("success.html", prediction=p, filename=img.filename)

    
if __name__ == "__main__":
    app.run()