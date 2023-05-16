import os
from distutils.dir_util import copy_tree
from shutil import rmtree

#Parameter
EXTRACTED_DIRECTORY = "./EXTRACTED"

#Iterriert Dateistruktur
for language in os.listdir(EXTRACTED_DIRECTORY):
    
    #Main Dir
    language_dir = os.path.join(EXTRACTED_DIRECTORY, language)

    #Select Main Dir
    dir = language_dir
    while True:
        folder_content = os.listdir(dir)
        
        #Falls clips Ordner gefunden wurde
        if "clips" in folder_content:
            break

        if len(folder_content) != 1:
            raise Exception("UNCLEAR PATH")
        dir = os.path.join(dir, folder_content[0])
    
    #Überspringt Ordner mit guter truktur
    if dir == language_dir:
        continue
    
    #Logging
    print(dir," -> ",language_dir)
    
    #Kopiert und löscht danach
    copy_tree(src = dir, dst = language_dir)
    rmtree(dir)