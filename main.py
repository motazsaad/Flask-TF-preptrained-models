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
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
classifier_model = model_url
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])
labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
#labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',)
imagenet_labels = np.array(open('ImageNetLabels.txt').read().splitlines())


@app.route('/classify-img1/')
def classify_img():
    url = ''
    if request.method == 'POST':
        url = request.form['url']
        # get image and preprocess (open, resize, convert to gray scale)
        response = requests.get(url)
        #grace_hopper = tf.keras.utils.get_file('./image.jpg', url)
        grace_hopper = Image.open(BytesIO(response.content)).resize(IMAGE_SHAPE)
        grace_hopper = np.array(grace_hopper)/255.0
        # classify 
        result = classifier.predict(grace_hopper[np.newaxis, ...])
        predicted_class = np.argmax(result[0], axis=-1)
        top_3_classes = np.argpartition(a, -3)[-3:]
        top_3_classes_labels = result[0][top_3_classes]
        #predicted_class_name = imagenet_labels[predicted_class]
        #return '<h1> prediction: {} </h1>'.format(predicted_class_name)
        return render_template('classify_img.html', zip(top_3_classes_labels, top_3_classes))

@app.route('/classify-img2/')
def classify_img2():
    url = request.args.get('url')
    # get image and preprocess (open, resize, convert to gray scale)
    response = requests.get(url)
    #grace_hopper = tf.keras.utils.get_file('./image.jpg', url)
    grace_hopper = Image.open(BytesIO(response.content)).resize(IMAGE_SHAPE)
    grace_hopper = np.array(grace_hopper)/255.0
    # classify 
    result = classifier.predict(grace_hopper[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)
    top_3_classes = np.argpartition(result[0], -3)[-3:]
    top_3_classes_labels = result[0][top_3_classes]
    top_3_classes_labels = [imagenet_labels[predicted_class] for predicted_class in top_3_classes_labels]
    #predicted_class_name = imagenet_labels[predicted_class]
    #return '<h1> prediction: {} </h1>'.format(predicted_class_name)
    # return render_template('classify_img2.html', predictions=zip(top_3_classes_labels, top_3_classes))
    return render_template('classify_img3.html', predicted_class=predicted_class)



# step 4: run app 
if __name__ == "__main__":  
  app.run(debug=True)

