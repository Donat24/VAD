import os
import tarfile

#PARAMETER
TAR_TRAIN = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/DATA/train.tar"
DIR_TRAIN = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/DATA/TRAIN"

TAR_TEST = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/DATA/test.tar"
DIR_TEST = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/DATA/TEST"

#Erzeugt Tar
for tar_file, dir in [(TAR_TRAIN, DIR_TRAIN), (TAR_TEST, DIR_TEST)]:

    #Löscht altes Tar
    if os.path.exists(tar_file):
        os.remove(tar_file)

    with tarfile.open(tar_file, mode="w") as tar:

        for file in os.listdir(dir):
            tar.add( name = os.path.join(dir, file), arcname=file)