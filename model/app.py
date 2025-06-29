import os
import uuid
import numpy as np
import cv2
from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import time
from werkzeug.utils import secure_filename
from tensorflow_addons.optimizers import AdamW


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

import tensorflow as tf 
from tensorflow.keras.layers import BatchNormalization, add, Conv2D, UpSampling2D, Resizing

kernel_initializer = 'he_uniform'

def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = K.cast(ground_truth, tf.float32)
    predictions = K.cast(predictions, tf.float32)
    ground_truth = K.flatten(ground_truth)
    predictions = K.flatten(predictions)
    intersection = K.sum(predictions * ground_truth)
    union = K.sum(predictions) + K.sum(ground_truth)

    dice = (2. * intersection + smooth) / (union + smooth)
    
    return (1 - dice)

def RAPU(x, filters): 
    x  = BatchNormalization(axis=-1)(x)
    
    x1 = atrous_block(x, filters)
    x2 = resnet_block(x, filters)
    
    x  = add([x1, x2])
    x  = BatchNormalization(axis=-1)(x)

    return x
 
    
def resnet_block(x, filters, dilation_rate=1):
    x1 = Conv2D(filters, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same',
                dilation_rate=dilation_rate)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x_final = add([x, x1])

    x_final = BatchNormalization(axis=-1)(x_final)

    return x_final    
    
def atrous_block(x, filters):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=1)(x)

    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=2)(x)

    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=3)(x)

    x = BatchNormalization(axis=-1)(x)

    return x
    
def convf_bn_act(inputs, filters, kernel_size, strides=(1, 1), activation='relu', padding='same'):
    
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    
    return x    
    
def SBA(L_input, H_input):
    dim = 16
    
    L_input = Conv2D(dim, 1, padding='same', use_bias=False)(L_input) 
    H_input = Conv2D(dim, 1, padding='same', use_bias=False)(H_input)    
      
    g_L = tf.keras.layers.Activation('sigmoid')(L_input)
    g_H = tf.keras.layers.Activation('sigmoid')(H_input)
    
    L_input = convf_bn_act(L_input, dim, 1) 
    H_input = convf_bn_act(H_input, dim, 1)   
    
    
    L_feature = L_input + L_input * g_L + (1 - g_L) * UpSampling2D((2,2))(g_H * H_input)
    H_feature = H_input + H_input * g_H + (1 - g_H) * Resizing(H_input.shape[1], H_input.shape[2])(g_L * L_input)
    
    H_feature = UpSampling2D((2,2))(H_feature)
    out = tf.keras.layers.Concatenate(axis=-1)([L_feature, H_feature])
    
    out = convf_bn_act(out, dim*2, 3)
    out = Conv2D(1, 1, use_bias=False)(out)
    
    return out

import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import add
from keras.models import Model
from keras_cv_attention_models import caformer

kernel_initializer = 'he_uniform'
interpolation = "nearest"

def unflatten_tokens(x, grid_size):
    """
    x: (batch, grid_size*grid_size, channels)
    returns: (batch, grid_size, grid_size, channels)
    """
    return tf.reshape(x, (-1, grid_size, grid_size, tf.shape(x)[-1]))

def create_model(img_height, img_width, input_chanels, out_classes, starting_filters):
    backbone = caformer.CAFormerS18(input_shape=(352, 352, 3), pretrained="imagenet", num_classes = 0)
    layer_names = ['stack4_block3_mlp_Dense_1', 'stack3_block9_mlp_Dense_1', 'stack2_block3_mlp_Dense_1', 'stack1_block3_mlp_Dense_1']
    layers =[backbone.get_layer(x).output for x in layer_names]
    
    input_layer = backbone.input
    print('Starting RAPUNet')

    p1 = Conv2D(starting_filters * 2, 3, strides=2, padding='same')(input_layer)  
  
    #from metaformer
    p2 = Conv2D(starting_filters*4, 1,padding='same')(unflatten_tokens(layers[3], grid_size=88))  # 88×88

    p3 = Conv2D(starting_filters*8,1,padding='same')(unflatten_tokens(layers[2], grid_size=44))  # 44×44

    p4 = Conv2D(starting_filters*16,1,padding='same')(unflatten_tokens(layers[1], grid_size=22))  # 22×22

    p5 = Conv2D(starting_filters*32,1,padding='same')(unflatten_tokens(layers[0], grid_size=11))  # 11×11
    
    
    t0 = RAPU(input_layer, starting_filters)  #352, 352

    # Encoder: RAPU + downsampling
    l1i = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(t0)    
    s1 = add([l1i, p1])     
    t1 = RAPU(s1, starting_filters * 2)  #176,176      
   
    
    l2i = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(t1)
    s2 = add([l2i, p2])
    t2 = RAPU(s2, starting_filters * 4) #88,88
    

    l3i = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(t2)
    s3 = add([l3i, p3])
    t3 = RAPU(s3, starting_filters * 8) #44,44
   
    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3)
    s4 = add([l4i, p4])
    t4 = RAPU(s4, starting_filters * 16) #22,22
    
    l5i = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(t4)
    s5 = add([l5i, p5]) 
    
    # bottleneck: ResNet Blocks 
    t51 = resnet_block(s5, starting_filters*32)
    t51 = resnet_block(t51, starting_filters*32) 
    t53 = resnet_block(t51, starting_filters*16) #11,11
    t53 = resnet_block(t53, starting_filters*16) #11,11

    # Save this "t53" as final encoder output
    encoder_output = t53
        
    #Aggregation

    dim =32
    
    outd = tf.keras.layers.Concatenate(axis=-1)([UpSampling2D((4,4))(t53), UpSampling2D((2,2))(t4)])
    outd = tf.keras.layers.Concatenate(axis=-1)([outd, t3]) #44,44 
    outd = convf_bn_act(outd, dim, 1)
    outd = Conv2D(1, kernel_size=1, use_bias=False)(outd)  #output1 44,44,1
    
    L_input = convf_bn_act(t2,dim, 3) #88,88,32   
    
    H_input = tf.keras.layers.Concatenate(axis=-1)([UpSampling2D((2,2))(t53), t4])
    H_input = convf_bn_act(H_input, dim,1)
    H_input = UpSampling2D((2,2))(H_input)  #44,44,32
    
    out2 = SBA(L_input, H_input) #output 88,88,1   
         
    out1 = UpSampling2D(size=(8,8), interpolation='bilinear')(outd)
    out2 = UpSampling2D(size=(4,4), interpolation='bilinear')(out2)
        
    out_duat = out1+out2 
    segmentation_output = Conv2D(out_classes,(1, 1),activation='sigmoid',name="segmentation_output")(out_duat)

    # Regression Model
    seg_down = AveragePooling2D(pool_size=(32,32))(segmentation_output)
    combined = tf.keras.layers.Concatenate(axis=-1)([encoder_output, seg_down])

    x = GlobalAveragePooling2D()(combined)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    regression_output = tf.keras.layers.Dense(4, activation='relu', name="regression_output")(x)
    
    model = Model(
        inputs=input_layer,
        outputs=[segmentation_output, regression_output]
    )

    return model


# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'alphapolyp_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload and result directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Global variable to store the model
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_ai_model():
    """Load the TensorFlow model with custom objects"""
    global model
    
    # Define custom objects for model loading
    custom_objects = {
        'AdamW': AdamW,
        'dice_metric_loss': dice_metric_loss
    }
    
    model_path = 'alphapolyp_optimized_model_3500cases.h5'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}", flush=True)
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model(model_path)
        print("Model loaded successfully!", flush=True)
    else:
        print(f"Model file {model_path} not found. Creating new model...", flush=True)
        model = create_model(352, 352, 3, 1, 17)
        print("Model created but not trained. Predictions will be random.", flush=True)

def preprocess_image(image_path, img_size=352):
    """Load and preprocess an image for prediction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}",flash=True)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def visualize_results(image_path, segmentation, volume, dimensions, subject_name):
    """Create visualization of the prediction results"""
    # Read original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}",flash=True)
    
    # Resize to match the model input size
    image = cv2.resize(image, (352, 352))
    
    # Create a copy of the original image
    original = image.copy()
    
    # Resize segmentation to match image dimensions
    segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]))
    
    # Create segmentation overlay with red color
    segmentation = (segmentation * 255).astype(np.uint8)
    
    # Create red overlay (BGR format)
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 2] = segmentation  # Red channel
    
    # Blend original image with red overlay
    overlay = cv2.addWeighted(image, 0.7, red_overlay, 0.3, 0)
    
    # Create a side-by-side display
    combined = np.hstack((original, overlay))
    
    return combined

@app.route('/')
def index():
    return "Model API is running!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the image
    preprocessed_img = preprocess_image(file_path)

    # Run the model prediction
    if model is not None:
        segmentation, regression = model.predict(preprocessed_img)
        segmentation = segmentation[0, :, :, 0].tolist()  # Convert to list for JSON
        volume = float(regression[0, 0])
        dimensions = [float(x) for x in regression[0, 1:4]]
        processing_time = 0  # You can add timing if you want

        return jsonify({
            'segmentation': segmentation,
            'volume': volume,
            'dimensions': dimensions,
            'processing_time': processing_time
        })
    else:
        return jsonify({'error': 'Model not loaded'}), 500

@app.route('/about')
def about():
    return render_template('about.html')

def pred_image(img_path, model, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
    
    pred = pred.squeeze().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    
    # Create output image with original and prediction side by side
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (256, 256))
    
    # Create a blank image for the prediction
    pred_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    pred_rgb[:, :, 0] = pred  # Red channel for the prediction
    
    # Combine original and prediction
    combined = np.hstack((original, pred_rgb))
    
    # Convert back to BGR for saving
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    
    return combined

# Load the AI model when the app starts (outside any function or if __name__ == '__main__')
try:
    load_ai_model()
except Exception as e:
    print(f"Error loading model: {str(e)}", flush=True)
    print("The application will use mock data for predictions.", flush=True)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)