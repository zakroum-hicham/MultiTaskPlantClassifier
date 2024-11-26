from flask import Flask, render_template, request,jsonify
import numpy as np
from PIL import Image
from io import BytesIO
from utils import predict
from werkzeug.utils import secure_filename
from utils import file_to_image
import os
import uuid

app = Flask(__name__)

@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/api/v1/classifier/v1",methods=['POST'])
def classifier():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    r = image_file.read()


    # predict with the model
    leaf_predicted,disease_predicted= predict(data=r,model_version="v1")
    # Return the predicted class and the confidence value
    if image_file:
            filename = secure_filename(image_file.filename)

            name = str(uuid.uuid4()) + os.path.splitext(filename)[1]

            cwd = os.getcwd()
            file_path = cwd+"\\static\\uploads\\"
            img = file_to_image(r).resize((256,256))
            img.save(file_path+name)
            return render_template('prediction.html',predicted_class=f"{leaf_predicted}",disease=f"{disease_predicted}",imagename=name)


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
