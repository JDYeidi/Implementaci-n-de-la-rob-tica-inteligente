from PIL import Image
from os import remove
#path = "/home/paul/VSCode/Python/DeepL/Manchester/Data/MyData/33/"
path = ""
contenido = os.listdir(path)
for fichero in contenido:
    if fichero.endswith('ppm'):
        nombre = fichero[:-4] + '.jpg'
        im = Image.open(path + fichero)
        im.save(path + nombre, quality=100)
        remove(path + fichero)