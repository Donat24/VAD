import os
import tarfile

#PARAMETER
TAR_TRAIN = "train.tar"
DIR_TRAIN = "./DATA/TRAIN"

TAR_TEST = "test.tar"
DIR_TEST = "./DATA/TEST"

#Erzeugt Tar
for tar_file, dir in [(TAR_TRAIN, DIR_TRAIN), (TAR_TEST, DIR_TEST)]:

    #Löscht altes Tar
    if os.path.exists(tar_file):
        os.remove(tar_file)

    with tarfile.open(tar_file, mode="w") as tar:

        for file in os.listdir(dir):
            tar.add( name = file )