import numpy as np
import keras

class SimpleDataFromNpzGenerator(keras.utils.Sequence):
    """
    Generate batches of data contained in npz files in a folder.
    Parameters
    ----------
    folder : string
        Folder where npz files are
    names_of_files : list of strings
        List of npz files in folder
    dim: tuple of int
        Dimension of data in npz
    name_of_data_in_npz : string, default='data'
        name for data in npz file
    name_of_label_in_npz : string, default='label'
        name for label in npz file
    batch_size : int, default=32
        Number of samples in output array of each iteration of the 'generate'
        method.
    shuffle : boolean, default=True
    """
    def __init__(self, 
                 npzfiles,
                 dim,
                 name_of_data_in_npz = 'data',
                 name_of_label_in_npz = 'label',
                 batch_size = 32,
                 shuffle = True):
        'Init'
        self.npzfiles = npzfiles
        self.dim = dim
        self.name_of_data_in_npz = name_of_data_in_npz
        self.name_of_label_in_npz = name_of_label_in_npz
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.npzfiles))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.npzfiles) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_npzfiles_temp = [self.npzfiles[k] for k in indexes]
        
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 2))
        
        # Generate data
        for i, ID in enumerate(list_npzfiles_temp):           
            npz = np.load(ID)
            X[i,] = npz[self.name_of_data_in_npz].reshape((128,173,1))
            y[i] = npz[self.name_of_label_in_npz][0]
            npz.close()

        return X, y