import os
import tarfile

#Parameter
TAR_DIRECTORY = "./COMMON VOICE"
OUT_DIRECTORY = "./EXTRACTED"

#Filecontent
tar_language_mapping = {
    "en.tar"                              : "english",
    "de.tar"                              : "german",
    "cv-corpus-13.0-2023-03-09-da.tar.gz" : "dutch",
    "fr.tar"                              : "french",
    "it.tar"                              : "italian",
    "es.tar"                              : "spanish",
    "pt.tar"                              : "portuguese",
    "cv-corpus-13.0-2023-03-09-el.tar.gz" : "greek",
    "cv-corpus-12.0-2022-12-07-hu.tar.gz" : "hungarian",
    "ru.tar"                              : "russian",
    "pl.tar"                              : "polish",
    "zh-CN.tar"                           : "chinese",
    "cv-corpus-8.0-2022-01-19-ja.tar.gz"  : "japanese",
    "cv-corpus-10.0-2022-07-04-fi.tar.gz" : "finnish",
    "ar.tar"                              : "arabic",
    "fa.tar"                              : "persian"
}

#Erstellt OUT_DIRECTORY
if not os.path.exists(OUT_DIRECTORY):
    os.mkdir(OUT_DIRECTORY)

#Iterriert Datein
for file in os.listdir(TAR_DIRECTORY):
    
    #Sprache
    language = tar_language_mapping[file]

    #Erstellt Ordner
    language_path = os.path.join(OUT_DIRECTORY, language)
    if not os.path.exists(language_path):
        os.mkdir(language_path)

    #Logging
    print(f"Bearbeite Datensatz '{language}'")

    #Öffnet TAR
    with tarfile.open(os.path.join(TAR_DIRECTORY, file)) as tar:
        tar.extractall(path=language_path)