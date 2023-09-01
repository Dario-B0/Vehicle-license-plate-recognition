import tensorflow as tf

def ious(true,pred):
    xt, yt, wt, ht = tf.split(true, 4, axis=-1)
    xp, yp, wp, hp = tf.split(pred, 4, axis=-1)

    xmin = tf.maximum(xt-(wt/2) ,xp-(wp/2) )
    ymin = tf.maximum(yt-(ht/2) ,yp-(hp/2) )
    xmax = tf.minimum(xt+(wt/2) ,xp+(wp/2) )
    ymax = tf.minimum(yt+(ht/2) ,yp+(hp/2) )

    intersection = tf.maximum(xmax - xmin, 0.) * tf.maximum(ymax - ymin, .0)

    union = (wt*ht) + (wp*hp) - intersection

    ious_= tf.math.divide_no_nan(intersection,union)

    return ious_


def CoordLoss(y_true, y_pred):

    l_o=1.

    #find if it exist an object in the grid
    existsObject = tf.expand_dims(y_true[..., 4], -1)

    xy_pred = existsObject * y_pred[..., 0:2]
    xy_true = existsObject * y_true[..., 0:2]
    wh_pred = existsObject * tf.sqrt(y_pred[..., 2:4])
    wh_true = existsObject * tf.sqrt(y_true[..., 2:4])

    coordLoss = tf.reduce_sum(tf.math.square(wh_pred - wh_true))
    coordLoss += (l_o * tf.reduce_sum(tf.math.square(xy_pred - xy_true)))

    return coordLoss

def ConfidenceLoss(y_true, y_pred):

    l_obj=5.
    l_no=1.

    existsObject = tf.expand_dims(y_true[..., 4], -1)

    iou_scores = ious(y_true[...,0:4],y_pred[...,0:4])

    best_ious = tf.reduce_max(iou_scores, axis=4)

    confidenceLoss = (l_obj * tf.reduce_sum(tf.math.square(existsObject * (iou_scores - y_pred[..., 4:5]))))
    confidenceLoss += (l_no * tf.reduce_sum(tf.math.square((1 - existsObject) * (0 - y_pred[...,4:5]))))

    return confidenceLoss


def yoloLoss(y_true, y_pred):

    coordLoss = CoordLoss(y_true, y_pred)
    confidenceLoss = ConfidenceLoss(y_true, y_pred)

    return  coordLoss  +  confidenceLoss

def metric_avg_iou(y_true, y_pred):

  IoUs = ious(y_true[...,0:4],y_pred[...,0:4])

  existsObject = tf.expand_dims(y_true[..., 4], -1)

  return  tf.reduce_sum(IoUs) / tf.cast(tf.math.count_nonzero(existsObject), dtype=tf.float32)