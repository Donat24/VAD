import os
import shutil

#Params
from_path = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/GENERIERUNG/AVA"
to_path   = r"/mnt/data/source_jonas/VAD/Scripte/data/samples/DATA/OTHER/AVA"

#Checkt Path
if not os.path.exists(to_path):
    os.mkdir(to_path)

#Copy Files
for file in os.listdir(from_path):
    
    #Check for Wav
    if file.endswith(".wav"):

        #Copy
        shutil.copy(
            src=os.path.join(from_path,file),
            dst=to_path
        )

        #Remove
        os.remove(os.path.join(from_path,file))

    
