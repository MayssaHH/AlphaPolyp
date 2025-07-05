from ast import Constant
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import add
from keras.models import Model
from keras_cv_attention_models import caformer
from model_architecture.RAPU_blocks import resnet_block, RAPU, convf_bn_act, SBA 

kernel_initializer = 'he_uniform'
interpolation = "nearest"


def create_model(img_height, img_width, input_channels, out_classes, starting_filters, reg_mean_norm=None):
    backbone = caformer.CAFormerS18(input_shape=(img_height, img_width, input_channels), pretrained="imagenet", num_classes = 0)
    layer_names = ['stack4_block3_mlp_Dense_1', 'stack3_block9_mlp_Dense_1', 'stack2_block3_mlp_Dense_1', 'stack1_block3_mlp_Dense_1']
    layers =[backbone.get_layer(x).output for x in layer_names]
    print(layers[1].shape)

    
    input_layer = backbone.input
    print('Starting RAPUNet')
    print(input_layer)
    p1 = Conv2D(starting_filters * 2, 3, strides=2, padding='same')(input_layer)  
    print("output shape of p1", p1.shape)
    #from metaformer
    p2 = Conv2D(starting_filters*4,1,padding='same')(layers[3]) 
    print("output shape of p2", p2.shape)   
    p3 = Conv2D(starting_filters*8,1,padding='same')(layers[2]) 
    print("output shape of p3", p3.shape)
    p4 = Conv2D(starting_filters*16,1,padding='same')(layers[1])     
    print("output shape of p4", p4.shape)
    p5 = Conv2D(starting_filters*32,1,padding='same')(layers[0]) 
    print("output shape of p5", p5.shape)
    
    
    t0 = RAPU(input_layer, starting_filters)  
    print("output shape of t0", t0.shape)
    # Encoder: RAPU + downsampling
    l1i = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(t0)    
    s1 = add([l1i, p1])     
    t1 = RAPU(s1, starting_filters * 2)       
    print("output shape of t1", t1.shape)
    
    l2i = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(t1)
    s2 = add([l2i, p2])
    t2 = RAPU(s2, starting_filters * 4) 
    print("output shape of t2", t2.shape)

    l3i = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(t2)
    s3 = add([l3i, p3])
    t3 = RAPU(s3, starting_filters * 8) 
    print("output shape of t3", t3.shape)
    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3)
    s4 = add([l4i, p4])
    t4 = RAPU(s4, starting_filters * 16)
    print("output shape of t4", t4.shape)
    l5i = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(t4)
    s5 = add([l5i, p5]) 
    print("output shape of s5", s5.shape)
    # bottleneck: ResNet Blocks 
    t51 = resnet_block(s5, starting_filters*32)
    t51 = resnet_block(t51, starting_filters*32) 
    t53 = resnet_block(t51, starting_filters*16) 
    t53 = resnet_block(t53, starting_filters*16) 
    print("output shape of t53", t53.shape)
    # Save this "t53" as final encoder output
    encoder_output = t53
        
    # Aggregation
    dim =32
    
    outd = tf.keras.layers.Concatenate(axis=-1)([UpSampling2D((4,4))(t53), UpSampling2D((2,2))(t4)])
    outd = tf.keras.layers.Concatenate(axis=-1)([outd, t3]) 
    outd = convf_bn_act(outd, dim, 1)
    outd = Conv2D(1, kernel_size=1, use_bias=False)(outd)  
    print("output shape of outd", outd.shape)
    L_input = convf_bn_act(t2,dim, 3)   
    print("output shape of L_input", L_input.shape)
    H_input = tf.keras.layers.Concatenate(axis=-1)([UpSampling2D((2,2))(t53), t4])
    H_input = convf_bn_act(H_input, dim,1)
    H_input = UpSampling2D((2,2))(H_input) 
    print("output shape of H_input", H_input.shape)
    out2 = SBA(L_input, H_input)   
    print("output shape of out2", out2.shape)
    out1 = UpSampling2D(size=(8,8), interpolation='bilinear')(outd)
    out2 = UpSampling2D(size=(4,4), interpolation='bilinear')(out2)
    print("output shape of out1", out1.shape)
    print("output shape of out2", out2.shape)
    out_duat = out1+out2 
    print("output shape of out_duat", out_duat.shape)
    segmentation_output = Conv2D(out_classes, (1, 1), activation='sigmoid')(out_duat)
    print("output shape of segmentation_output", segmentation_output.shape)
    # Regression Model
    seg_down = AveragePooling2D(pool_size=(32,32))(segmentation_output)
    combined = tf.keras.layers.Concatenate(axis=-1)([encoder_output, seg_down])
    print("output shape of combined", combined.shape)
    x = GlobalAveragePooling2D()(combined)
    print("output shape of x", x.shape)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    print("output shape of x", x.shape)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    print("output shape of x", x.shape)

    # Initialize the bias of the regression output layer
    bias_init = Constant(reg_mean_norm) if reg_mean_norm is not None else "zeros"

    regression_output = tf.keras.layers.Dense(4, activation='relu', bias_initializer=bias_init, name="regression_output")(x)
    
    model = Model(
        inputs=input_layer,
        outputs=[segmentation_output, regression_output]
    )

    return model
