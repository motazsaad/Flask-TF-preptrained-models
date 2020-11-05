# step 1: import Flask 
from flask import Flask, render_template, request

 

from io import BytesIO
import requests
import numpy as np 
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub


# step 2: define Flask app 
app = Flask(__name__)


@app.route('/')
def home():
  return render_template('home.html')


##################################
IMAGE_SHAPE = (224, 224)

mobilenet_v2 = tf.keras.Sequential([
    hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", 
    input_shape=IMAGE_SHAPE+(3,))
])
# resnet_50 = tf.keras.Sequential([
    # hub.KerasLayer(
    # "https://tfhub.dev/tensorflow/resnet_50/classification/1", 
    # input_shape=IMAGE_SHAPE+(3,))
# ])

inception_v3 = tf.keras.Sequential([
    hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/inception_v3/classification/4", 
    input_shape=IMAGE_SHAPE+(3,))
])


labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
#labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',)
imagenet_labels = np.array(open('ImageNetLabels.txt').read().splitlines())

# models = {'mobilenet_v2': mobilenet_v2, 'resnet_50': resnet_50, 'inception_v3': inception_v3}
models = {'mobilenet_v2': mobilenet_v2, 'inception_v3': inception_v3}

def get_labels(image):
    results = {}
    for model_name, model in models.items():
        preds = model.predict(image[np.newaxis, ...])
        predicted_class = np.argmax(preds[0], axis=-1)
        label = imagenet_labels[predicted_class]
        results[model_name] = label
    
    return results 
    

@app.route('/classify-img1/', methods=['GET', 'POST'])
def classify_img():
    url = ''
    top_labels = {}
    if request.method == 'POST':
        url = request.form['url']
        if url:
            print('url', url)
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).resize(IMAGE_SHAPE)
            img = np.array(img)/255.0
            # classify 
            result = mobilenet_v2.predict(img[np.newaxis, ...])
            cert =  np.exp(result[0])/sum(np.exp(result[0]))
            predicted_class = np.argmax(result[0], axis=-1)
            label = imagenet_labels[predicted_class]
            top_index = result[0].argsort()[-3:][::-1]
            top_labels = { imagenet_labels[i]: cert[i] for i in top_index }
    
    print(top_labels)
    return render_template('classify_img1.html', top_labels=top_labels, url=url)

@app.route('/classify-img2/')
def classify_img2():
    url = request.args.get('url')
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).resize(IMAGE_SHAPE)
    img = np.array(img)/255.0
    # classify 
    result = mobilenet_v2.predict(img[np.newaxis, ...])
    cert =  np.exp(result[0])/sum(np.exp(result[0]))
    predicted_class = np.argmax(result[0], axis=-1)
    label = imagenet_labels[predicted_class]
    top_index = result[0].argsort()[-3:][::-1]
    top_labels = { imagenet_labels[i]: cert[i] for i in top_index }
    print(top_labels)
    return render_template('classify_img2.html', top_labels=top_labels)


@app.route('/classify-img3/', methods=['GET', 'POST'])
def classify_img3():
    url = ''
    labels = {}
    if request.method == 'POST':
        url = request.form['url']
        if url:
            print('url', url)
            response = requests.get(url)
            print('get image', response.status_code)
            img = Image.open(BytesIO(response.content)).resize(IMAGE_SHAPE)
            img = np.array(img)/255.0
            print(img.shape)
            # classify 
            labels = get_labels(img)
    
    print(labels)
    return render_template('classify_img3.html', labels=labels, url=url)

# step 4: run app 
if __name__ == "__main__":  
  app.run()

