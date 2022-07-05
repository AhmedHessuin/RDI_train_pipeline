import tensorflow as tf
import os
import numpy as np
import glob
import cv2
from logger import logger
import datetime
from model import model_builder,compile_model
from typing import Callable, Any, Iterable, Generator
from os import path

logger.info("main run ")

def train_env(gpu_device:str="0"):
    '''
    set the config of what will train gpu or cpu
    :param gpu_device: string, the number of gpu "number" or "" for cpu
    :return:
    '''
    logger.debug(gpu_device)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    logger.info("starting training on gpu no: "+gpu_device)



def call_backs(base_line_active:bool=False, baseline:float=0.05):
    logdir = os.path.join("train_log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    return [tensorboard_callback]


def get_images(image_path:str):
    '''
    get any image end with valid extension for the model
    :param image_path: string contain the path to the directory contain images
    :return: list contain the abs paths for the images in the given dir
    '''
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(image_path, '*.{}'.format(ext))))
    return files

def generator(training_data_path:str,
              batch_size:int, vis:bool=False):
    '''
    custom data generator support online data loading and labeling  taking the data path and the batch
    size and yield a numpy list contain the batch size of images and it's GT
    :param training_data_path: string, contain the data path
    :param batch_size: int, contain the batch size
    :return: yield batches to the fit functions
    '''

    logdir = os.path.join("train_log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    image_list = np.array(get_images(training_data_path))
    logger.info('{} training images in {}'.format(
        image_list.shape[0], training_data_path))

    index = np.arange(0, image_list.shape[0])

    while True:
        np.random.shuffle(index)
        images = []
        labels=[]
        for i in index:
            try:
                im_fn = image_list[i]
                img=cv2.imread(im_fn)
                h,w,_=img.shape
                img=cv2.resize(img,dsize=(150,150))
                if "NORMAL" in im_fn:
                    label=[0,1]
                else:
                    label=[1,0]

                images.append(img)
                labels.append(label)
                if vis:
                    cv2.imshow("img",img)
                    logger.debug(label)
                    cv2.waitKey()
                if len(images)==batch_size:
                    images = np.array(images)
                    labels = np.array(labels)
                    yield [images],[labels]
                    images=[]
                    labels=[]
            except:
                logger.exception("image: ",image_list[i]," contain an error check "
                                                     "the path if path exist, it "
                                                     "cv2 reading error,image corrupted")


def train(model:tf.keras.Model, epochs:int, batch:int, data_path:str, val_path:str,
          generator:Callable[[str,int,bool],
                             None],
          call_backs:Callable[[bool,float],list]):
    '''
    full pipline to train a model and return history
    :param model: tf.keras.Model. the model
    :param epochs: int, number of epochs
    :param batch: int, number of batches
    :param data_path: string, train data path
    :param generator: function, train data generator
    :param val_path : string, the path to the validation data
    :param call_backs: function, callback functions
    :return: history
    '''


    logger.info("train started")
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(150, 150),
        batch_size=batch,
        label_mode="categorical"
        )

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    history=model.fit(generator(training_data_path=data_path,batch_size=batch),
              steps_per_epoch=len(get_images(data_path))//batch, # best 16  batch                size for bw
        # best for color is batch size 8
        epochs=epochs, #625 101 for 5000 , 128 for 1025
        use_multiprocessing=True,
        workers=24,
        callbacks=call_backs(),
        max_queue_size=50,
        verbose=1,
        validation_data=val_ds)
    logger.info(f"train finished with history{history.history} ")

    check_point_path="./checkpoints/"
    for model_number in range(10000):
        if not path.isdir(f'./checkpoints/{model_number}'):
            check_point_path=check_point_path+f"{model_number}"
            break

    os.makedirs(check_point_path, exist_ok=True)
    model.save(check_point_path)
    return history

logger.info("train env")
train_env("0")
logger.info("model build")
model=model_builder(150)
model=compile_model(model)
train(model,4,32,"Data/Train","Data/test",generator,call_backs)



