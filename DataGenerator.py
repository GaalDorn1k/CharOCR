import numpy as np
import cv2
import os


class DataGenerator():
    """
    Генератор ТОЛЬКО для торчевых моделей
    генерирует:
    кропы: shape (batch_size, 3, h, w)
    маски: shape (batch_size, len(alphabet)+1, h, w)
    """
    def __init__(self, img_size: tuple, batch_size: int,
                 img_path: str, charmask_path: str, fieldmask_path: str, shuffle=True) -> None:
        self.img_size = img_size
        self.batch_size = batch_size
        self.img_names = os.listdir(img_path)
        self.charmask_path = charmask_path
        self.fieldmask_path = fieldmask_path
        self.img_path = img_path
        cm = np.load(os.path.join(self.charmask_path, self.img_names[0].replace('png', 'npy')))
        self.alphabet_len = cm.shape[1]
        fm = np.load(os.path.join(self.charmask_path, self.img_names[0].replace('png', 'npy')))
        self.fields_num = fm.shape[1]
        self.shuffle = shuffle
        self.list_IDs = self.img_names
        self.__on_epoch_end()

    def __on_epoch_end(self) -> None:
        self.indexes = np.arange(len(self.img_names))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, 3, *self.img_size))
        y = np.empty((self.batch_size, self.alphabet_len, *self.img_size))
        y2 = np.empty((self.batch_size, self.fields_num, *self.img_size))

        for i, name in enumerate(list_IDs_temp):
            img = cv2.imread(os.path.join(self.img_path, name))
            # img = cv2.resize(img, self.img_size)
            img = img.swapaxes(0, 2)
            img = img.swapaxes(1, 2)
            img = img / 255.0
            X[i, ] = img
            char_mask = np.load(os.path.join(self.charmask_path, name.replace('png', 'npy')))
            field_mask = np.load(os.path.join(self.fieldmask_path, name.replace('png', 'npy')))
            # resize_mask = []

            # for layer in char_mask:
            #     # layer = cv2.resize(layer, self.img_size)
            #     resize_mask.extend([layer])
            # resize_mask = np.array(resize_mask)
            y[i, ] = char_mask
            y2[i, ] = field_mask

        return X, y, y2

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[i] for i in indexes]
        X, y, y2 = self.__data_generation(list_IDs_temp)
        return X, y, y2

    def __len__(self) -> int:
        return len(self.list_IDs) // self.batch_size
