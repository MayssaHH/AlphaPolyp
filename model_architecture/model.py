import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import add
from keras.models import Model
from keras_cv_attention_models import caformer
from model_architecture.RAPU_blocks import resnet_block, RAPU, convf_bn_act, SBA

kernel_initializer = 'he_uniform'
interpolation = "nearest"

# CAFormerS18 outputs tokens of shape (batch, H*W, C) from its MLP Dense layers.
# This helper reshapes them to (batch, H, W, C) so Conv2D can process them.
def unflatten_tokens(x, grid_h, grid_w=None):
    if grid_w is None:
        grid_w = grid_h
    channels = tf.shape(x)[-1]
    return tf.reshape(x, (-1, grid_h, grid_w, channels))


def create_model(img_height=352, img_width=352, input_channels=3,
                 out_classes=1, starting_filters=17, bias=None):
    """
    Build the AlphaPolyp model: CAFormerS18 backbone + RAPUNet decoder
    for segmentation, plus a regression head for (logVolume, x, y, z).

    img_height / img_width must both be 352 (backbone pretrained at that size).
    starting_filters is the base filter count (paper default: 17).
    bias: optional array of shape (4,) used to initialise the regression head bias
          (typically set to the training-set mean of [logVol, x, y, z]).
    """
    assert img_height == 352 and img_width == 352, (
        f"CAFormerS18 is pretrained on 352×352 input. Got {img_height}×{img_width}."
    )

    backbone = caformer.CAFormerS18(
        input_shape=(img_height, img_width, input_channels),
        pretrained="imagenet",
        num_classes=0
    )
    # Named MLP Dense outputs from each stage.
    # All four are tokens: (batch, H*W, C) where H depends on the stage's downsampling.
    layer_names = [
        'stack4_block3_mlp_Dense_1',   # 32× down → 11×11
        'stack3_block9_mlp_Dense_1',   # 16× down → 22×22
        'stack2_block3_mlp_Dense_1',   # 8×  down → 44×44
        'stack1_block3_mlp_Dense_1',   # 4×  down → 88×88
    ]
    layers = [backbone.get_layer(x).output for x in layer_names]

    # Grid sizes at each stage for 352×352 input
    g4 = img_height // 32   # 11
    g3 = img_height // 16   # 22
    g2 = img_height // 8    # 44
    g1 = img_height // 4    # 88

    input_layer = backbone.input

    # Stem path (from raw input)
    p1 = Conv2D(starting_filters * 2, 3, strides=2, padding='same')(input_layer)  # 176×176

    # Backbone feature projections — unflatten tokens before Conv2D
    p2 = Conv2D(starting_filters *  4, 1, padding='same')(unflatten_tokens(layers[3], g1))  # 88×88
    p3 = Conv2D(starting_filters *  8, 1, padding='same')(unflatten_tokens(layers[2], g2))  # 44×44
    p4 = Conv2D(starting_filters * 16, 1, padding='same')(unflatten_tokens(layers[1], g3))  # 22×22
    p5 = Conv2D(starting_filters * 32, 1, padding='same')(unflatten_tokens(layers[0], g4))  # 11×11

    # Encoder
    t0 = RAPU(input_layer, starting_filters)                            # 352×352

    l1i = Conv2D(starting_filters * 2,  2, strides=2, padding='same')(t0)
    s1  = add([l1i, p1])
    t1  = RAPU(s1, starting_filters * 2)                                # 176×176

    l2i = Conv2D(starting_filters * 4,  2, strides=2, padding='same')(t1)
    s2  = add([l2i, p2])
    t2  = RAPU(s2, starting_filters * 4)                                # 88×88

    l3i = Conv2D(starting_filters * 8,  2, strides=2, padding='same')(t2)
    s3  = add([l3i, p3])
    t3  = RAPU(s3, starting_filters * 8)                                # 44×44

    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3)
    s4  = add([l4i, p4])
    t4  = RAPU(s4, starting_filters * 16)                               # 22×22

    l5i = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(t4)
    s5  = add([l5i, p5])                                                # 11×11

    # Bottleneck
    t51 = resnet_block(s5,  starting_filters * 32)
    t51 = resnet_block(t51, starting_filters * 32)
    t53 = resnet_block(t51, starting_filters * 16)
    t53 = resnet_block(t53, starting_filters * 16)
    encoder_output = t53                                                 # 11×11

    # Decoder — coarse path
    dim  = 32
    outd = tf.keras.layers.Concatenate(axis=-1)([UpSampling2D((4, 4))(t53), UpSampling2D((2, 2))(t4)])
    outd = tf.keras.layers.Concatenate(axis=-1)([outd, t3])             # 44×44
    outd = convf_bn_act(outd, dim, 1)
    outd = Conv2D(1, kernel_size=1, use_bias=False)(outd)

    # Decoder — fine path (SBA)
    L_input = convf_bn_act(t2, dim, 3)                                  # 88×88
    H_input = tf.keras.layers.Concatenate(axis=-1)([UpSampling2D((2, 2))(t53), t4])
    H_input = convf_bn_act(H_input, dim, 1)
    H_input = UpSampling2D((2, 2))(H_input)                             # 44×44 → 44×44 (stays)

    out2 = SBA(L_input, H_input)
    out1 = UpSampling2D(size=(8, 8), interpolation='bilinear')(outd)    # 44→352
    out2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(out2)    # 88→352

    out_duat = out1 + out2
    segmentation_output = Conv2D(
        out_classes, (1, 1), activation='sigmoid', name="segmentation_output"
    )(out_duat)

    # Regression head — fuse encoder features with a downsampled segmentation mask
    seg_down = AveragePooling2D(pool_size=(32, 32))(segmentation_output)  # 11×11×1
    combined = tf.keras.layers.Concatenate(axis=-1)([encoder_output, seg_down])

    x = GlobalAveragePooling2D()(combined)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Bias init: warm-start with training-set mean so the head starts near the target range.
    # NOTE: ReLU clips negatives; logVolume mean should be positive after log1p transform.
    bias_init = tf.keras.initializers.Constant(bias) if bias is not None else "zeros"
    regression_output = tf.keras.layers.Dense(
        4, activation='linear', bias_initializer=bias_init, name="regression_output"
    )(x)

    model = Model(inputs=input_layer, outputs=[segmentation_output, regression_output])
    return model
