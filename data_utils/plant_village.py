import os
import sys
import numpy as np
import tensorflow as tf

# path to PlantVillage dataset containing train and test directories or plant**.npz for 32,64,and 96
base_path = "../githubs/myrepos/downsampled-plant-disease-dataset/data/"


def count(path, counter=0):
    "returns number of files in dir and subdirs"
    for pack in os.walk(path):
        for f in pack[2]:
            counter += 1
    return counter


def plant_village_scratch(name):
    """
    Loads original PlantVillage dataset once and converts to the required shape and saves as numpy array
    :param name: name with resolution size
    :return: Training and test images and labels
    """
    # data_path = base_path+'datasets/plant-disease'
    validation_dir = base_path + '/test'
    train_dir = base_path + '/train'
    print('Dataset not found reading and writing from scratch train_dir=', train_dir)
    val_size = count(validation_dir)
    train_size = count(train_dir)
    if '32' in name:
        pixels = 32
    elif '64' in name:
        pixels = 64
    elif '96' in name:
        pixels = 96
    else:
        print("invalid dataset name")
        sys.exit(1)
    image_size = (pixels, pixels)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(validation_dir, color_mode="rgb", class_mode='sparse',
                                                                  target_size=image_size, batch_size=val_size)
    train_generator = validation_datagen.flow_from_directory(train_dir, color_mode="rgb", target_size=image_size,
                                                             batch_size=train_size, class_mode='sparse')
    train_images, train_labels = train_generator.next()
    test_images, test_labels = validation_generator.next()
    print('saving ', base_path+'/' + name + '.npz', 'with pixels ', pixels)
    np.savez_compressed(base_path+'/' + name + '.npz', train_images=train_images, train_labels=train_labels,
                        test_images=test_images, test_labels=test_labels)
    return train_images, train_labels, test_images, test_labels


def load_dataset(name):
    """
    Loads PlantVillage dataset.
    :param name: name of the dataset, one of plant32, plant64, plant96
    :return: Training and test images and labels
    """
    file_path = base_path + name + '.npz'
    if os.path.exists(file_path):
        npzfile = np.load(file_path)
        train_images, train_labels,  = npzfile['train_images'], npzfile['train_labels']
        test_images, test_labels = npzfile['test_images'], npzfile['test_labels']
        return train_images, train_labels, test_images, test_labels
    return plant_village_scratch(name)
