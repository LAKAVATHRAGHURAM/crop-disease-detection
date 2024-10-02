import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/plant_details/apple')
def apple():
    return render_template('plant_details/apple.html')

@app.route('/plant_details/blueberry')
def blueberry():
    return render_template('plant_details/blueberry.html')

@app.route('/plant_details/cherry')
def cherry():
    return render_template('plant_details/cherry.html')

@app.route('/plant_details/corn')
def corn():
    return render_template('plant_details/corn.html')

@app.route('/plant_details/grape')
def grape():
    return render_template('plant_details/grape.html')

@app.route('/plant_details/orange')
def orange():
    return render_template('plant_details/orange.html')

@app.route('/plant_details/peach')
def peach():
    return render_template('plant_details/peach.html')

@app.route('/plant_details/pepparbell')
def pepparbell():
    return render_template('plant_details/pepparbell.html')

@app.route('/plant_details/potato')
def potato():
    return render_template('plant_details/potato.html')

@app.route('/plant_details/raspberry')
def raspberry():
    return render_template('plant_details/raspberry.html')

@app.route('/plant_details/soybean')
def soybean():
    return render_template('plant_details/soybean.html')

@app.route('/plant_details/squash')
def squash():
    return render_template('plant_details/squash.html')

@app.route('/plant_details/strawberry')
def strawberry():
    return render_template('plant_details/strawberry.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
