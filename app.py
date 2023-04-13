from flask import Flask, render_template, url_for, request
import sqlite3
import shutil
import os
import sys
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from twilio.rest import Client
account_sid = "ACe4cd5f8d160458bf1470987cfe969db7"
auth_token = "447e21d0b0bbecb236a1123cee9bcdb6"
client = Client(account_sid, auth_token)

import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('userlog.html')

@app.route('/leaf')
def leaf():
    return render_template('leaf.html')

@app.route('/leaf_disease', methods=['GET', 'POST'])
def leaf_disease():
    if request.method == 'POST':
        image = request.form['img']

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'leafdisease-{}-{}.model'.format(LR, '2conv-basic')

        def process_verify_data():
            verifying_data = []
            path = 'static/test/'+image
            img_num = image.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 6, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        # fig = plt.figure()
        diseasename=" "
        rem=" "
        rem1=" "
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            # y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Healthy'
            elif np.argmax(model_out) == 1:
                str_label = 'Bacterial'
            elif np.argmax(model_out) == 2:
                str_label = 'mosaic_virus'
            elif np.argmax(model_out) == 3:
                str_label = 'Spectoria'
            elif np.argmax(model_out) == 4:
                str_label = 'leafmold'
            elif np.argmax(model_out) == 5:
                str_label = 'curl_virus'

            if str_label == 'Bacterial':
                diseasename = "Bacterial Spot "
                rem = "The remedies for Bacterial Spot are:\n\n "
                rem1 = [" Discard or destroy any affected plants",  
                "Do not compost them.", 
                "Rotate yoour tomato plants yearly to prevent re-infection next year.", 
                "Use copper fungicites"]
                client.api.account.messages.create(
                                                to="+91-7993557128",
                                                from_="+15077065830",
                                                body="Bacterial  disease, copper fungicites")



            if str_label == 'curl_virus':
                diseasename = "Yellow leaf curl virus "
                rem = "The remedies for Yellow leaf curl virus are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates", 
                "carbametes during the seedliing stage.", "Use copper fungicites"]
                client.api.account.messages.create(
                                                to="+91-7993557128",
                                                from_="+15077065830",
                                                body="curl_virus  disease, organophosphates and carbametes")

                
            if str_label == 'Spectoria':
                diseasename = "Spectoria "
                rem = "The remedies for Spectoria are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"]  
                client.api.account.messages.create(
                                                to="+91-7993557128",
                                                from_="+15077065830",
                                                body="Spectoria  disease, organophosphates and carbametes")
    

            if str_label == 'leafmold':
                diseasename = "leafmold"
                rem = "The remedies for Leafmold are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
                client.api.account.messages.create(
                                                to="+91-7993557128",
                                                from_="+15077065830",
                                                body="Leafmold  disease, chemical fungicides(chlorothalonil)")

            if str_label == 'Healthy':
                status = "Healthy"
                client.api.account.messages.create(
                                                to="+91-7993557128",
                                                from_="+15077065830",
                                                body="Healthy")
    

            if str_label == 'mosaic_virus':
                diseasename = "mosaic_virus"

                rem = "The remedies for mosaic_virus are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"] 
                client.api.account.messages.create(
                                                to="+91-7993557128",
                                                from_="+15077065830",
                                                body="mosaic_virus disease, carbametes")
    

        return render_template('leaf.html', status=str_label, disease=diseasename, remedie=rem, remedie1=rem1, ImageDisplay="http://127.0.0.1:5000/static/test/"+image)

    return render_template('leaf.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
