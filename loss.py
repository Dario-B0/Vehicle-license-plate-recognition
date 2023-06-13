import tensorflow as tf

def IoUs(target,predictionA,predictionB):
    x_t,y_t,w_t,h_t=target[...,0],target[...,1],target[...,2],target[...,3]
    x_A,y_A,w_A,h_A=predictionA[...,0],predictionA[...,1],predictionA[...,2],predictionA[...,3]
    x_B,y_B,w_B,h_B=predictionB[...,0],predictionB[...,1],predictionB[...,2],predictionB[...,3]

    #conversion in order to compute intesetctions and unions
    x_tmin=x_t - (w_t / 2)
    x_tmax=x_t + (w_t / 2)
    y_tmin=y_t - (h_t / 2)
    y_tmax=y_t + (h_t / 2)

    x_Amin=x_A - (w_A / 2)
    x_Amax=x_A + (w_A / 2)
    y_Amin=y_A - (h_A / 2)
    y_Amax=y_A + (h_A / 2)
    
    x_Bmin=x_B - (w_B / 2)
    x_Bmax=x_B + (w_B / 2)
    y_Bmin=y_B - (h_B / 2)
    y_Bmax=y_B + (h_B / 2)

    #areas of the bounding box of target and predictions
    bb_A=tf.maximum((w_A + 1)*(h_A +1),0)
    bb_B=tf.maximum((w_B + 1)*(h_B +1),0)
    bb_t=(w_t + 1)*(h_t +1)         #should never be lower than zero

    #for bbA a compute the corner of intersection
    x_minIA=tf.maximum(x_tmin,x_Amin)  #xA
    x_maxIA=tf.minimum(x_tmax,x_Amax)  #xB
    y_minIA=tf.maximum(y_tmin,y_Amin)  #yA
    y_maxIA=tf.minimum(y_tmax,y_Amax)  #yB

    #intersection
    intA=tf.maximum(0,x_maxIA - x_minIA + 1) * tf.maximum(0,y_maxIA - y_minIA + 1)

    #union  
    uniA= bb_A + bb_t - intA

    iouA=intA / uniA


    #for bbB a compute the corner of intersection
    x_minIB=tf.maximum(x_tmin,x_Bmin)  #xA
    x_maxIB=tf.minimum(x_tmax,x_Bmax)  #xB
    y_minIB=tf.maximum(y_tmin,y_Bmin)  #yA
    y_maxIB=tf.minimum(y_tmax,y_Bmax)  #yB

    #intersection
    intB=tf.maximum((x_maxIB - x_minIB + 1),0) * tf.maximum((y_maxIB - y_minIB + 1),0)

    #union  
    uniB= bb_B + bb_t - intB

    iouB=intB / uniB

    return tf.stack([iouA,iouB])


def YOLOloss(target,predictions):
    G=7  #  GRID 
    B=2  #  bb per cell 

    ious = IoUs(target[..., 1:5],predictions[..., 1:5],predictions[..., 6:])  # ... x 7 x 7 x 2
    #print(ious)
    best_pred=tf.expand_dims(tf.cast(tf.argmax(ious),float),3)
    #print(best_pred)

    best_ious=tf.expand_dims(tf.maximum(ious[0],ious[1]),3)
    #print(best_ious)

    is_box=tf.expand_dims(target[...,0],3)
    #print(is_box)

    #print(predictions[..., 1:5])

    ###box coordinates error
    #take only the best box (higher IoU) in order to compute the error

    best_bb= is_box * ( best_pred * predictions[..., 6:] + (1 - best_pred) * predictions[..., 1:5])
    #print(best_bb)

    target_bb=is_box*target[..., 1:5]
    #print(target_bb)

    lambda_coord=5.
    loss_bb_coord=lambda_coord*tf.reduce_sum(tf.pow(best_bb[...,0]-target_bb[...,0],2)+tf.pow(best_bb[...,1]-target_bb[...,1],2))
    #print(loss_bb_coord)


    ###box dimension error
    loss_bb_dim=lambda_coord*tf.reduce_sum(tf.pow(tf.sqrt(best_bb[...,2])-tf.sqrt(target_bb[...,2]),2)+tf.pow(tf.sqrt(best_bb[...,3])-tf.sqrt(target_bb[...,3]),2))
    #print(loss_bb_dim)

    predictions=tf.expand_dims(predictions,3)
    ###object confidence error
    best_bb_confidence= is_box * (best_pred * predictions[..., 5] + (1 - best_pred)*predictions[..., 0])
    #print(best_bb_confidence)

    loss_obj_conf=tf.reduce_sum(tf.pow((is_box*best_ious)-best_bb_confidence,2))
    #print(loss_obj_conf)

    

    ###on-object confidence error
    best_noobj_confidence= (1-is_box) * (best_pred * predictions[..., 5] + (1 - best_pred)*predictions[..., 0])
    #print(best_noobj_confidence)

    lambda_noobj=0.5
    #print(is_box)
    #print(best_ious)
    #print((1-is_box)*best_ious)
    loss_noobj_conf=lambda_noobj*tf.reduce_sum(tf.pow(((1-is_box)*best_ious)-best_noobj_confidence,2))
    #print(loss_noobj_conf)

    return loss_bb_coord + loss_bb_dim + loss_obj_conf + loss_noobj_conf