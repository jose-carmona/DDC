import glob

def npz_in_folder(folder):
    return glob.glob(folder + '/*.npz')