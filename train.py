#!/usr/bin/python3

from functools import reduce
import cv2
import numpy as np
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from pycocotools.coco import COCO
from keras.metrics import binary_accuracy
from random import shuffle
from scipy.misc import imresize
import threading
import random
import keras.backend as K
import tensorflow as tf
from src import masknet
from keras.utils import multi_gpu_model


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def roi_pool_cpu(frame, bbox, pool_size):
    frame_h, frame_w = frame.shape[:2]

    x1 = int(bbox[0] * frame_w)
    y1 = int(bbox[1] * frame_h)
    w1 = int(bbox[2] * frame_w)
    h1 = int(bbox[3] * frame_h)

    slc = frame[y1:y1+h1, x1:x1+w1, ...]

    if (w1 <= 0) or (h1 <= 0):
        assert(np.count_nonzero(slc) == 0)
        return slc

    slc = imresize(slc.astype(float), (pool_size, pool_size), 'nearest') / 255.0

    return slc


def process_coco(coco, img_path, limit):
    """Process COCO/COCO like dataset to generate list of tuples which contains bbox, roi, msks.
    """
    res = []
    #cat_ids1 = coco.getCatIds(catNms=['person'])
    #cat_ids2 = coco.getCatIds(catNms=['backpack'])
    #img_ids1 = coco.getImgIds(catIds=cat_ids1)
    #img_ids2 = coco.getImgIds(catIds=cat_ids2)
    #img_ids = list(set(img_ids1).union(set(img_ids2)))
    cat_ids = coco.getCatIds()
    print(cat_ids)
    img_ids = coco.getImgIds(catIds=cat_ids[0])
    print(len(img_ids))
    imgs = coco.loadImgs(ids=img_ids)
    processed = 0
    iter1 = 0

    # fake_msk = np.zeros((masknet.my_msk_inp * 2, masknet.my_msk_inp * 2), dtype=np.uint8).astype('float32')

    if limit:
        imgs = imgs[:limit]

    for img in imgs:
        iter1 += 1
        processed += 1
        if iter1 > 1000:  # report every 1000 imgs
            iter1 = 0
            print("processed", processed, '/', len(imgs))

        ann_ids = coco.getAnnIds(imgIds=img['id'], areaRng=[1024, 2073600])
        anns = coco.loadAnns(ann_ids)
        frame_w = img['width']
        frame_h = img['height']
        rois = []
        msks = []
        bboxs = []
        cocos = []
        for ann in anns:
            if ('bbox' in ann) and (ann['bbox'] != []) and ('segmentation' in ann):
                bbox = [int(xx) for xx in ann['bbox']]
                bbox[0] /= frame_w
                bbox[1] /= frame_h
                bbox[2] /= frame_w
                bbox[3] /= frame_h

                # m = coco.annToMask(ann)

                # msk = roi_pool_cpu(m, bbox, masknet.my_msk_inp * 2)

                assert(len(rois) < masknet.my_num_rois)

                x1 = np.float32(bbox[0])
                y1 = np.float32(bbox[1])
                w1 = np.float32(bbox[2])
                h1 = np.float32(bbox[3])

                rois.append([y1, x1, y1 + h1, x1 + w1])
                msks.append(ann)
                bboxs.append(bbox)
                cocos.append(coco)
        if len(rois) > 0:
            for _ in range(masknet.my_num_rois - len(rois)):
                rois.append([np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)])
                msks.append(None)
                bboxs.append(None)
                cocos.append(None)
            res.append((img['file_name'], img_path, rois, msks, bboxs, cocos))

    return res


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def my_preprocess(im):
    h = 320
    w = 320
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return imsz


@threadsafe_generator
def fit_generator(imgs, batch_size):
    ii = 0
    fake_msk = np.zeros((masknet.my_msk_inp * 2, masknet.my_msk_inp * 2), dtype=np.uint8).astype('float32')
    while True:
        shuffle(imgs)
        for m in range(len(imgs) // batch_size):
            i = m * batch_size
            j = i + batch_size
            if j > len(imgs):
                j = - j % len(imgs)
            batch = imgs[i:j]
            x1 = []
            x2 = []
            y = []
            for img_name, img_path, rois, anns, bboxs, cocos in batch:
                # x12.append(np.load("masknet_data_17/" + img_name.replace('.jpg', '.npz'))['arr_0'])
                # x13.append(np.load("masknet_data_28/" + img_name.replace('.jpg', '.npz'))['arr_0'])
                # x14.append(np.load("masknet_data_43/" + img_name.replace('.jpg', '.npz'))['arr_0'])

                # flip = random.randint(0, 1) == 0
                # flip = False

                frame = cv2.imread(img_path + "/" + img_name)
                # if flip:
                #     frame = np.fliplr(frame)
                x1.append(my_preprocess(frame))

                my_rois = []
                for roi in rois:
                    rx1 = roi[1]
                    rx2 = roi[3]
                    # if flip:
                    #     rx1 = 1.0 - roi[3]
                    #     rx2 = 1.0 - roi[1]
                    my_rois.append([roi[0], rx1, roi[2], rx2])

                x2.append(np.array(my_rois))

                msks = []
                for n in range(len(bboxs)):
                    if cocos[n] is None:
                        msk = fake_msk
                    else:
                        msk = roi_pool_cpu(cocos[n].annToMask(anns[n]), bboxs[n], masknet.my_msk_inp * 2)
                        # if flip:
                        #     msk = np.fliplr(msk)
                    msks.append(msk)

                msks = np.array(msks)
                msks = msks[..., np.newaxis]

                y.append(msks)
            # gc.collect()
            # print("yield",ii)
            ii += 1
            yield ([np.array(x1), np.array(x2)], np.array(y))


def my_accuracy(y_true, y_pred):
    mask_shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))
    mask_shape = tf.shape(y_true)
    y_true = K.reshape(y_true, (-1, mask_shape[2], mask_shape[3], mask_shape[4]))

    sm = tf.reduce_sum(y_true, [1, 2, 3])

    ix = tf.where(sm > 0)[:, 0]

    y_true = tf.gather(y_true, ix)
    y_pred = tf.gather(y_pred, ix)

    return binary_accuracy(y_true, y_pred)


def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet53(x):
    '''Darknent body having 52 Convolution2D layers'''

    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x1 = resblock_body(x, 128, 2)
    x2 = resblock_body(x1, 256, 8)
    x3 = resblock_body(x2, 512, 8)
    x4 = resblock_body(x3, 1024, 4)
    return x1, x2, x3, x4


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = Input(shape=(320, 320, 3))
    x1, x2, x3, x4 = darknet53(inputs)
    darknet = Model(inputs, x4)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1, y2, y3]), inputs, x1, x2, x3, x4


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        tmp_model = self.model
        self.model = self.my_model
        super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.model = tmp_model


if __name__ == "__main__":

    # generate yolo model and it's feature maps
    with tf.device('/cpu:0'):
        yolo_model, yolo_input, yolo_C2, yolo_C3, yolo_C4, yolo_C5 = yolo_body(3, 2)
        yolo_model.load_weights('yolov3_best_final.h5')
        for i in range(252):
            layers = yolo_model.layers[i]
            layers.trainable = False

    # generate masknet model and it's output tensor
        mn_model = masknet.create_model()
        #mn_model.load_weights('mask14_best.hdf5')
        mn_model.summary()
        m_roi_input = Input(shape=(masknet.my_num_rois, 4))
        x = mn_model([yolo_C2, yolo_C3, yolo_C4, yolo_C5, m_roi_input])

    # generate model and compile
        model = Model(inputs=[yolo_input, m_roi_input], outputs=x)
        
        model.summary()
    parallel_model = multi_gpu_model(model, gpus=3)
    parallel_model.compile(loss=[masknet.my_loss], optimizer='adam', metrics=[my_accuracy])

    # process coco dataset
    #bdir = '/home/exjobb/COCO2014'
    #train_coco = COCO(bdir + "/annotations/instances_train2014.json")
    #val_coco = COCO(bdir + "/annotations/instances_val2014.json")
    #train_imgs = process_coco(train_coco, bdir + "/images/train2014", None)
    #val_imgs = process_coco(val_coco, bdir + "/images/val2014", None)

    #train_coco = None
    #val_coco = None

    #train_imgs += val_imgs[5000:]
    #val_imgs = val_imgs[:5000]

    # fine-tine dataset
    bdir = '/home/exjobb/golfer_data'
    First1_coco = COCO(bdir + '/First1.json')
    First2_coco = COCO(bdir + '/First2.json')
    Second_coco = COCO(bdir + '/Second.json')
    Third_coco = COCO(bdir + '/Third.json')
    First1_imgs = process_coco(First1_coco, bdir + '/First', None)
    First2_imgs = process_coco(First2_coco, bdir + '/First', None)
    Second_imgs = process_coco(Second_coco, bdir + '/Second', None)
    Third_imgs = process_coco(Third_coco, bdir + '/Third', None)

    First1_coco = None
    First2_coco = None
    Second_coco = None
    Third_coco = None
    
    train_imgs = First1_imgs + First2_imgs + Second_imgs + Third_imgs
    #train_imgs = First1_imgs
    val_imgs = train_imgs


    

    batch_size = 48

    # prepare train/validation input for the network
    train_data = fit_generator(train_imgs, batch_size)
    validation_data = fit_generator(val_imgs, batch_size)

    # lr_schedule = lambda epoch: 0.001 if epoch < 120 else 0.0001
    lr_schedule = lambda epoch: 0.001
    # lr_schedule = lambda epoch: 1e-5
    callbacks = [LearningRateScheduler(lr_schedule)]

    mcp = MyModelCheckpoint(filepath="mask14_direct.hdf5", monitor='val_loss', save_best_only=True)
    mcp.my_model = mn_model
    callbacks.append(mcp)

    parallel_model.fit_generator(train_data,
                        steps_per_epoch=len(train_imgs) / batch_size,
                        validation_steps=len(val_imgs) / batch_size,
                        epochs=100,
                        validation_data=validation_data,
                        max_queue_size=10,
                        workers=1, use_multiprocessing=False,
                        verbose=1,
                        callbacks=callbacks)
    print("Done!")
