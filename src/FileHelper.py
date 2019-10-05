from pathlib import Path

def getFilenameWithoutExtension(filepath):
    fullPath = Path(filepath)
    filename = fullPath.parts[-1]
    index = filename.find('.')
    filenameWithoutExt = filename[0:index]
    return filenameWithoutExt 
