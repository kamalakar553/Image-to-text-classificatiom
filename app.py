from flask import Flask, render_template, request, redirect, url_for
import pytesseract
import cv2
import joblib
from PIL import Image
import io
import re
import os
import numpy as np

app = Flask(__name__, template_folder='template')
# configurations
config = ('-l eng --oem 1 --psm 3')
pytesseract.pytesseract.tesseract_cmd = r'Downloads\Tesseract'
model = joblib.load(open('model.pkl', 'rb'))

def draw_box_nd_save(img):
    if os.path.exists("static/img.png"):
        os.remove("static/img.png")
    h,w =img.size
    cnfg= "--psm 11 --oem 3"
    boxes=pytesseract.image_to_boxes(img,config=config)
    for box in boxes.splitlines():
        box=box.split(" ")
        img=cv2.rectangle(np.array(img),(int(box[1]),h-int(box[2])),(int(box[3]),h-int(box[4])),(0,0,255),2)
    cv2.imwrite("static/img.png",img)





@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY) # Converting img to gray 
        text = pytesseract.image_to_string(gray,config=config) # Apply OCR to extract text
        text = re.sub("\n", " ", text)
        pred=model.predict([text])
        draw_box_nd_save(img)
        print(text)
        return render_template("index.html", res=1,pred=pred[0],text=text)

    return render_template("index.html", res=0)


if __name__ == "__main__":
    app.run(debug=True)
