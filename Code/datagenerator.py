import numpy as np
import tensorflow as tf

from NetworkBasis import config as cfg
import NetworkBasis.loadsavenii as loadsave

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, filenames, path, nb,shape):
        self.filenames = filenames
        self.path = path
        self.nb = nb
        self.shape=shape

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.nb))).astype(np.int)

    def __getitem__(self, idx):
        batch = self.filenames[idx * self.nb: (idx + 1) * self.nb]
        vol_shape = self.shape
        ndims = len(self.shape)
        zero_phi = np.zeros([1, *vol_shape, ndims])

        images=[]
        for file_name in batch:
            image=loadsave.load_image(self.path, file_name[0][2:-1])
            images.append(image)

        images = np.array(images)
        images = np.moveaxis(images, 0, -1)

        if cfg.print_details:
            print("images.shape", images.shape)

        images = np.expand_dims(images, axis=0)
        inputs = images

        template=np.mean(images, -1)
        template = np.expand_dims(template, axis=-1)

        outputs=[]
        for i in range(self.nb):
            outputs.append(template)
            outputs.append(zero_phi)

        return (inputs, outputs)

def get_test_images(filenames, path, idx):

    batch = filenames[idx * cfg.nb: (idx + 1) * cfg.nb]

    images = []
    for file_name in batch:
        image = loadsave.load_image(path, file_name[0][2:-1])
        images.append(image)

    images = np.array(images)
    images = np.moveaxis(images, 0, -1)

    images = np.expand_dims(images, axis=0)
    inputs = images

    return (inputs)
