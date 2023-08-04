from flask import Flask, render_template, request
import numpy as np
import os
import numpy as np
import argparse

import cv2
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("C:/Users/karti/Desktop/projects/OCR/OCR.tf")
prediction_model=keras.models.Model(model.get_layer(name='image').input,
                                    model.get_layer(name='dense2').output)
symbols=['A','C','D','I','M','P','R','T','V','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
from tensorflow.keras import layers
char_to_num=layers.StringLookup(vocabulary=symbols, 
                                mask_token=None)
num_to_char=layers.StringLookup(vocabulary=char_to_num.get_vocabulary(),
                                mask_token=None, 
                                invert=True)
print("model is loaded")

app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        # img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(img_path)         
        img_path="C:/Users/karti/Pictures/Screenshots/2.png"
        img=tf.io.read_file(img_path)
        img=tf.io.decode_png(img,1)
        print(img.shape)
        IMAGE_HEIGHT=32
        IMAGE_WIDTH=128
        img=tf.image.resize_with_pad(img,target_width=IMAGE_WIDTH,target_height=IMAGE_HEIGHT)
        img=tf.transpose(img,perm=[1,0,2])
        print(img.shape)
        img=tf.cast(img,dtype=tf.float32)
        img=np.expand_dims(img,axis=0)
        img=img/255.0

        pred = prediction_model.predict(img)
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     print(input_len)
        result = keras.backend.ctc_decode(pred, 
                                       input_length=input_len, 
                                       greedy=True)
        result=result[0][0][:, :14]
        output_text = []
        for res in result:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        print("output",output_text[0])
        passed_path=img_path[1:-1]
        return render_template('sec.html', pred_output=output_text, user_image=img_path)

if __name__ == "__main__":
    app.run(threaded=False)