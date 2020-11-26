import numpy as np 
import tensorflow as tf 
import keras
from keras import backend as K
from keras import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from keras import optimizers

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
from ipdb import set_trace

def swp_loss(label,pred):
    #js beta
    G = label.shape[1]
    G_blood =tf.reduce_sum(tf.gather_nd(label,tf.where(label>=1)))
    # G = label.shape[1]*label.shape[2]
    # G_blood = label[label>0].sum()
    beta = (G_blood)/G
    pred = tf.reshape(pred,(pred.get_shape()[0],pred.get_shape()[1]*pred.get_shape()[2]))
    # pred = pred.reshape(pred.shape[0],pred.shape[1]*pred.shape[2])

    sigle_swp_loss = -beta*(tf.reduce_sum(tf.math.log(tf.gather_nd(pred,tf.where(label>=1)))))-(1-beta)*(tf.reduce_sum(tf.math.log(tf.gather_nd(pred,tf.where(label<=0)))))
    # sigle_swp_loss = -beta*(np.log(pred[np.where(label>0)]).sum())-(1-beta)*(np.log(pred[np.where(label<0)]).sum())
    return sigle_swp_loss

def swp_sum_loss(label,pred_list):
    label = tf.reshape(label,(label.get_shape()[1],label.get_shape()[2]*label.get_shape()[3]))
    lf_loss = tf.zeros((1))
    for i in pred_list:
        if i != 0:
            temp = swp_loss(label,i)
            lf_loss += temp
    return lf_loss

def distance_loss_liux(label,pred_list,s_weight_list=["liux",0.2,0.2,0.2,0.2,0.2]):
    # print(K.eval(pred_list))
    print(tf.print(label))
    lf_loss = swp_sum_loss(label,pred_list)
    # print(lf_loss)
    zero_array = tf.zeros((1,244,244))
    for i in range(len(pred_list)):
        if i == 0:
            continue
        else:
            temp = s_weight_list[i]*i
            zero_array+=temp
    ls_loss = np.sqrt(((zero_array-label)**2).sum())
    sum_loss = lf_loss+ls_loss
    return sum_loss

def create_vsscnet(n_ch,patch_height,patch_width):
    #====================================================================================================
    #vgg16前4卷积层
    inputs = keras.Input(shape=(n_ch,patch_height,patch_width),name="img")
    conv1 = layers.Conv2D(64,3,activation="relu",padding="same",data_format="channels_first",name="1-1C")(inputs)
    conv1 = layers.Conv2D(64,3,activation="relu",padding="same",data_format="channels_first",name="1-2C")(conv1)
    pool1 = layers.MaxPooling2D((2,2),data_format="channels_first")(conv1)

    conv2 = layers.Conv2D(128,3,activation="relu",padding="same",data_format="channels_first",name="2-1C")(pool1)
    conv2 = layers.Conv2D(128,3,activation="relu",padding="same",data_format="channels_first",name="2-2C")(conv2)
    pool2 = layers.MaxPooling2D((2,2),data_format="channels_first")(conv2)

    conv3 = layers.Conv2D(256,3,activation="relu",padding="same",data_format="channels_first",name="3-1C")(pool2)
    conv3 = layers.Conv2D(256,3,activation="relu",padding="same",data_format="channels_first",name="3-2C")(conv3)
    conv3 = layers.Conv2D(256,3,activation="relu",padding="same",data_format="channels_first",name="3-3C")(conv3)
    pool3 = layers.MaxPooling2D((2,2),data_format="channels_first")(conv3)

    conv4 = layers.Conv2D(512,3,activation="relu",padding="same",data_format="channels_first",name="4-1C")(pool3)
    conv4 = layers.Conv2D(512,3,activation="relu",padding="same",data_format="channels_first",name="4-2C")(conv4)
    conv4 = layers.Conv2D(512,3,activation="relu",padding="same",data_format="channels_first",name="4-3C")(conv4)

    #====================================================================================================
    #vsc_1
    vsc_1_1 = layers.Conv2D(8,1,activation="relu",strides=1 ,data_format="channels_first",name="vsc-1-1")(conv1)
    vsc_1_2 = layers.Conv2D(8,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-1-2")(vsc_1_1)
    vsc_1_3 = layers.Conv2D(8,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-1-3")(vsc_1_2)
    vsc_1_4 = layers.Conv2D(8,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-1-4")(vsc_1_3)

    #skip
    x = layers.add([vsc_1_1,vsc_1_3])
    x = layers.Conv2D(8,1,activation="relu",padding="same",data_format="channels_first",name="vsc1-skip-conv-1")(x)
    x = layers.Conv2D(8,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc1-skip-conv-2")(x)

    x = layers.add([vsc_1_2,vsc_1_4,x])
    x = layers.Conv2D(8,1,activation="relu",padding="same",data_format="channels_first",name="vsc1-skip-conv-3")(x)
    sc1_output = layers.Conv2D(8,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc1-skip-conv-4")(x)

    #=====================================================================================================
    #vsc_2
    vsc_2_1 = layers.Conv2D(16,1,activation="relu",padding="same",data_format="channels_first",name="vsc-2-1")(conv2)
    vsc_2_2 = layers.Conv2D(16,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-2-2")(vsc_2_1)
    vsc_2_3 = layers.Conv2D(16,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-2-3")(vsc_2_2)
    vsc_2_4 = layers.Conv2D(16,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-2-4")(vsc_2_3)

    #skip
    sc1_downsampling = layers.Conv2D(16,3,strides=2,padding="same",activation="relu",data_format="channels_first",name="sc1_down")(sc1_output)
    x = layers.add([vsc_2_1,vsc_2_3,sc1_downsampling])
    x = layers.Conv2D(16,1,activation="relu",padding="same",data_format="channels_first",name="vsc2-skip-conv-1")(x)
    x = layers.Conv2D(16,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc2-skip-conv-2")(x)
    x = layers.add([vsc_2_2,vsc_2_4,x])
    x = layers.Conv2D(16,1,activation="relu",padding="same",data_format="channels_first",name="vsc2-skip-conv-3")(x)
    sc2_output = layers.Conv2D(16,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc2-skip-conv-4")(x)

    #=======================================================================================================
    #vsc_3
    vsc_3_1 = layers.Conv2D(32,1,activation="relu",padding="same",data_format="channels_first",name="vsc-3-1")(conv3)
    vsc_3_2 = layers.Conv2D(32,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-3-2")(vsc_3_1)
    vsc_3_3 = layers.Conv2D(32,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-3-3")(vsc_3_2)
    vsc_3_4 = layers.Conv2D(32,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-3-4")(vsc_3_3)

    #skip
    sc2_downsampling= layers.Conv2D(32,3,strides=2,padding="same",activation="relu",data_format="channels_first",name="sc2_down")(sc2_output)
    x = layers.add([vsc_3_1,vsc_3_3,sc2_downsampling])
    x = layers.Conv2D(32,1,activation="relu",padding="same",data_format="channels_first",name="vsc3-skip-conv-1")(x)
    x = layers.Conv2D(32,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc3-skip-conv-2")(x)
    x = layers.add([vsc_3_2,vsc_3_4,x])
    x = layers.Conv2D(32,1,activation="relu",padding="same",data_format="channels_first",name="vsc3-skip-conv-3")(x)
    sc3_output = layers.Conv2D(32,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc3-skip-conv-4")(x)

    # =======================================================================================================
    #vsc_4
    vsc_4_1 = layers.Conv2D(64,1,activation="relu",padding="same",data_format="channels_first",name="vsc-4-1")(conv4)
    vsc_4_2 = layers.Conv2D(64,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-4-2")(vsc_4_1)
    vsc_4_3 = layers.Conv2D(64,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-4-3")(vsc_4_2)
    vsc_4_4 = layers.Conv2D(64,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc-4-4")(vsc_4_3)

    #skip
    sc3_downsampling= layers.Conv2D(64,3,strides=2,padding="same",activation="relu",data_format="channels_first",name="sc3_down")(sc3_output)
    x = layers.add([vsc_4_1,vsc_4_3,sc3_downsampling])
    x = layers.Conv2D(64,1,activation="relu",padding="same",data_format="channels_first",name="vsc4-skip-conv-1")(x)
    x = layers.Conv2D(64,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc4-skip-conv-2")(x)
    x = layers.add([vsc_4_2,vsc_4_4,x])
    x = layers.Conv2D(64,1,activation="relu",padding="same",data_format="channels_first",name="vsc4-skip-conv-3")(x)
    sc4_output = layers.Conv2D(64,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc4-skip-conv-4")(x)

    # =======================================================================================================================================
    #to 4 fearture map
    v1 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(sc1_output)
    x2 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(sc2_output)
    x3 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(sc3_output)
    x4 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(sc4_output)
    #upsampeling to input image
    # x1 = layers.UpSampling2D((2,2),data_format="channels_first")(x1)
    v2 = layers.UpSampling2D((2,2),data_format="channels_first")(x2)
    v3 = layers.UpSampling2D((4,4),data_format="channels_first")(x3)
    v4 = layers.UpSampling2D((8,8),data_format="channels_first")(x4)
    #=========================================================================================================================================
    #to 4 fearture map
    f1 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(conv1)
    conv_2 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(conv2)
    conv_3 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(conv3)
    conv_4 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(conv4)
    #upsampeling to input image
    # conv_1 = layers.UpSampling2D((2,2),data_format="channels_first")(conv_1)
    f2 = layers.UpSampling2D((2,2),data_format="channels_first")(conv_2)
    f3 = layers.UpSampling2D((4,4),data_format="channels_first")(conv_3)
    f4 = layers.UpSampling2D((8,8),data_format="channels_first")(conv_4)
    #============================================================================================================================================
    #sum
    s1 = layers.add([f1,v2,v3,v4,f4])
    s2 = layers.add([f2,v1,v3,v4,f4])
    s3 = layers.add([f3,v1,v2,v4,f4])
    vsc5_input = s1
    vsc6_input = layers.MaxPooling2D((2,2),strides=2,data_format="channels_first")(s2)
    vsc7_input = layers.MaxPooling2D((4,4),strides=4,data_format="channels_first")(s3)

    vsc5_input = layers.Conv2D(8,1,activation="relu",padding="same",data_format="channels_first")(vsc5_input)
    vsc6_input = layers.Conv2D(16,1,activation="relu",padding="same",data_format="channels_first")(vsc6_input)
    vsc7_input = layers.Conv2D(32,1,activation="relu",padding="same",data_format="channels_first")(vsc7_input)


    #================================================================================================================================================
    vsc7_1 = vsc7_input
    vsc7_2 = layers.Conv2D(32,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc7-1")(vsc7_1)
    vsc7_3 = layers.Conv2D(32,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc7-2")(vsc7_2)
    vsc7_4 = layers.Conv2D(32,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc7-3")(vsc7_3)
    
    x = layers.add([vsc7_1,vsc7_3])
    x = layers.Conv2D(32,1,activation="relu",padding="same",data_format="channels_first",name="vsc7-4")(x)
    x = layers.Conv2D(32,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc7-5")(x)
    x = layers.add([vsc7_2,vsc7_4,x])
    x = layers.Conv2D(32,1,activation="relu",padding="same",data_format="channels_first",name="vsc7-6")(x)
    vsc7_output = layers.Conv2D(32,3,kernel_initializer="random_normal",activation="relu",padding="same",data_format="channels_first",name="vsc7-7")(x)
    x = layers.UpSampling2D((2,2),data_format="channels_first")(vsc7_output)
    vsc7_to_6 = layers.Conv2D(16,1,activation="relu",padding="same",data_format="channels_first",name="vsc7-8")(x)

    #====================================================================================================================================================
    vsc6_1 = vsc6_input
    vsc6_2 = layers.Conv2D(16,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc6-1")(vsc6_1)
    vsc6_3 = layers.Conv2D(16,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc6-2")(vsc6_2)
    vsc6_4 = layers.Conv2D(16,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc6-3")(vsc6_3)

    x = layers.add([vsc6_1,vsc6_3,vsc7_to_6])
    x = layers.Conv2D(16,1,activation="relu",padding="same",data_format="channels_first",name="vsc6-4")(x)
    x = layers.Conv2D(16,3,kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc6-5")(x)
    x = layers.add([vsc6_2,vsc6_4,x])
    x = layers.Conv2D(16,1,activation="relu",padding="same",data_format="channels_first",name="vsc6-6")(x)
    vsc6_output = layers.Conv2D(16,3,kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc6-7")(x)
    x = layers.UpSampling2D((2,2),data_format="channels_first")(vsc6_output)
    vsc6_to_5 = layers.Conv2D(8,1,activation="relu",padding="same",data_format="channels_first",name="vsc6-8")(x)
    
    #======================================================================================================================================================
    vsc5_1 = vsc5_input
    vsc5_2 = layers.Conv2D(8,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc5-1")(vsc5_1)
    vsc5_3 = layers.Conv2D(8,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc5-2")(vsc5_2)
    vsc5_4 = layers.Conv2D(8,3,activation="relu",kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc5-3")(vsc5_3)

    x = layers.add([vsc5_1,vsc5_3,vsc6_to_5])
    x = layers.Conv2D(8,1,activation="relu",padding="same",data_format="channels_first",name="vsc5-4")(x)
    x = layers.Conv2D(8,3,kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc5-5")(x)
    x = layers.add([vsc5_2,vsc5_4,x])
    x = layers.Conv2D(8,1,activation="relu",padding="same",data_format="channels_first",name="vsc5-6")(x)
    vsc5_output = layers.Conv2D(8,3,kernel_initializer="random_normal",padding="same",data_format="channels_first",name="vsc5-7")(x)
    x = layers.UpSampling2D((2,2),data_format="channels_first")(vsc5_output)

    #========================================================================================================================================================
    #to 4 map fearture
    vsc5_output_4 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(vsc5_output)
    vsc6_output_4 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(vsc6_output)
    vsc7_output_4 = layers.Conv2D(4,1,activation="relu",padding="same",data_format="channels_first")(vsc7_output)
    #to input image
    f7 = vsc5_output_4 
    f6 = layers.UpSampling2D((2,2),data_format="channels_first")(vsc6_output_4)
    f5 = layers.UpSampling2D((4,4),data_format="channels_first")(vsc7_output_4)


    s4 = layers.add([f5,v2,v3,v4,f7])
    s5 = layers.add([f6,v1,v3,v4,f7])

    cancate_layer = layers.concatenate([s1,s2,s3,s4,s5],axis=1)

    conv_temp = layers.Conv2D(2,1,activation="relu",padding="same",data_format="channels_first")(cancate_layer)
    conv_temp = layers.Reshape((2,patch_height*patch_width))(conv_temp)
    conv_temp = layers.Permute((2,1))(conv_temp)
    output = layers.Activation("softmax")(conv_temp) 
    

    


    #=========================================================================================================================================================
    #to one feartrue map
    # s1_1 = layers.Conv2D(1,1,padding="same",activation="sigmoid",data_format="channels_first",name="liux1")(s1)
    # s2_1 = layers.Conv2D(1,1,padding="same",activation="sigmoid",data_format="channels_first",name="liux2")(s2)
    # s3_1 = layers.Conv2D(1,1,padding="same",activation="sigmoid",data_format="channels_first",name="liux3")(s3)
    # s4_1 = layers.Conv2D(1,1,padding="same",activation="sigmoid",data_format="channels_first",name="liux4")(s4)
    # s5_1 = layers.Conv2D(1,1,padding="same",activation="sigmoid",data_format="channels_first",name="liux5")(s5)

    
    #========================================================================================================================================================
    model = keras.Model(inputs=inputs,outputs=output)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # feature_extractor = keras.Model(inputs=model.inputs,outputs=[layer.output for layer in model.layers][-5:])
    model.compile(optimizer=sgd, loss=custom_loss, metrics=['accuracy'])
    return model

def custom_loss(y_true, y_pred,axis=-1):
    y_true = ops.convert_to_tensor_v2(y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    print(y_pred)
    y_pred = y_pred / math_ops.reduce_sum(y_pred, axis, True)
    epsilon = tf.convert_to_tensor(1e-07, y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon, 1. - epsilon)
    temp = y_true * math_ops.log(y_pred)
    temp = temp*[0.55,4.76]
    return -math_ops.reduce_sum(temp, axis)

# model,feature_extractor = create_vsscnet(1,48,48)
# model.compile(optimizer='sgd', loss={"liux1":custom_loss,"liux2":custom_loss,"liux3":custom_loss,"liux4":custom_loss,"liux5":custom_loss}, 
#     loss_weights=[0.2,0.2,0.2,0.2,0.2],metrics=['accuracy'])


# model = create_vsscnet(1,48,48)
# x = np.random.randint(255,size=(1,1,48,48)).astype("float32")
# label = np.random.randint(2,size=(1,1,48,48)).astype("float32")
# y = model.predict(x)
# set_trace()
# model.fit(x, label, epochs=1, batch_size=1, verbose=1)
# set_trace()
    


