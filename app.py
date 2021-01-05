import numpy as np
import os

from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from wtforms import SubmitField


base_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(base_dir, 'static')


photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

# Load the model:
cnn_model = load_model('food_baseline_model.h5')
CLASS_INDICES = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
       'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
       'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake',
       'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
       'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon',
       'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari',
       'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
       'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream',
       'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels',
       'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho',
       'pizza', 'pork_chop', 'poutine', 'prime_ri', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
       'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
       'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki',
       'tiramisu', 'tuna_tartare', 'waffles']


# Form where image will be uploaded:
class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file to upload!')])
    submit = SubmitField('Get Prediction')


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html', form=UploadForm(), results={}, filename="")


@app.route('/prediction/', methods=['POST'])
def prediction():
    # Saving file to folder
    file = request.files['photo']
    filename = secure_filename(file.filename)
    file.save(os.path.join('static', filename))

    results = return_prediction(filename=filename)
    return render_template('home.html', form=UploadForm(), results=zip(results[0], results[1]), filename=filename)


def return_prediction(filename):
    input_image_matrix = _image_process(filename)
    score = cnn_model.predict(input_image_matrix)
    class_index = cnn_model.predict_classes(input_image_matrix, batch_size=1)
    n=10
    top_n = score[0].argsort()[::-1][:n]
    percentage =np.sort(score[0])[::-1]
    labs = []
    perc = []
    for i in range(n):
        labs.append(CLASS_INDICES[top_n[i]])

    return labs, percentage


def _image_process(filename):
    img = image.load_img('static/' + filename, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    input_matrix = np.vstack([x])
    input_matrix /= 255.
    return input_matrix


if __name__ == '__main__':
    app.run(debug=True)
