import keras.backend as K
import tensorflow as tf

def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = K.cast(ground_truth, tf.float32)
    predictions = K.cast(predictions, tf.float32)
    ground_truth = K.flatten(ground_truth)
    predictions = K.flatten(predictions)
    intersection = K.sum(predictions * ground_truth)
    union = K.sum(predictions) + K.sum(ground_truth)

    dice = (2. * intersection + smooth) / (union + smooth)
    
    return (1 - dice)
    
def normalized_mse_loss(reg_stats, smooth=1e-6):
    reg_min   = tf.constant(reg_stats["min"],   dtype="float32")
    reg_range = tf.constant(reg_stats["range"], dtype="float32") + smooth

    def normalized_mse(y_true, y_pred):
        y_true_norm = (y_true - reg_min) / reg_range
        y_pred_norm = (y_pred - reg_min) / reg_range
        return tf.reduce_mean(tf.square(y_true_norm - y_pred_norm), axis=-1)
    
    return normalized_mse