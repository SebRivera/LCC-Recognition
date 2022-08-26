# Medium
Se desarrolló una entrada en [Medium](https://medium.com/@jesusmaing/lcc-recognition-7b2b77a8716f) explicando el funcionamiento y desarrollo del proyecto.

# Reconocimiento facial
El proyecto utiliza tensorflow y opencv para hacer reconocimiento facial

La aplicación intenta buscar un rostro en la webcam. Posteriormente, se toma una captura de ese frame y se manda a una red neuronal profunda para intentar buscar coincidencia de alguna foto registrada en la carpeta "ids"
![Alt Text](./presentacion/Media1.gif)


## ¿Qué librerias necesito para el funcionamiento?
En el repositorio se deja un archivo requeriments.txt en donde están las librerías utilizadas
Estas se pueden instalar utilizando pip install -r requirements.txt

*   OpenCv
*   Tensorflow
*   Scikit-learn

## ¿Cómo surgió este proyecto?
Surgió debido a la inspiración de querer poner a practica los conocimientos adquiridos en la materia de Inteligencia Artificial y Reconocimiento de patrones en la licenciatura en Ciencias de la Computación. La idea es hacer un proyecto colaborativo, en donde los estudiantes puedan ir mejorando el sistema con el pasar de las generaciones para poder tener un sistema de reconocimiento más preciso y sostificado.

## ¿Qué modelo se utilizó?
Para entrenar la inteligencia artificial se utilizó un modelo construido por google: [facenet](https://github.com/davidsandberg/facenet).

Para detectar una rostro se utilizó una red convolucional multitarea [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).

Para la clasificación de ID se utilizó una [Inception Resnet](https://arxiv.org/abs/1602.07261)

Para una mejor precisión, se utilizó una red convolucional ya entrenada con millones de imagenes ya clasificadas, este modelo se puede encontrar [aquí](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk)

## ¿Cómo lo pongo en funcionamiento?

* Se tiene que descargar el [modelo de facenet ya preentrenado](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk).
* Crear una carpeta llamada `ids` donde tendrá subcarpetas con los nombres de las personas que quieres reconocer.
* Instalar todas las dependencias necesarias.
* Importar la base de datos de MYSQL
* Modificar la base datos en el código LCCAplication.py, y ajustarlo a sus necesidades, se puede cambiar por sqlite

La carpeta tiene que de quedar así:

```LCC_RECOGNITION
├── detect_and_align.py
├── main.py
├── README.md
├── requirements.txt
├── model
│   ├── 20170512-110547.pb
|   ├── model-20170512-110547.ckpt-250000.data-00000-of-00001
|   ├── model-20170512-110547.ckpt-250000.index
|   ├── model-20170512-110547.meta
├── det1
│   ├── det1.npy
│   ├── det2.npy
│   ├── det3.npy
├── ids
│   ├── 219205955
│   │   ├── martin.png
│   │   ├── martin2.png
│   ├── 219219494
│   │   ├── sebas.png
│   │   ├── sebas2.png
```
Habrá más archivos de los que aparecen en el árbol presentado anteriorme. Deben de ir en la raíz de la carpeta
## Creadores
* Jesus Martin Garcia Encinas - https://github.com/Jesusmaing
* Sebastian Guadalupe Rivera de la Cruz - https://github.com/SebRivera
* Marco Antonio Guerrero Vasquez - https://github.com/Squisix
* Jesus Israel Urias Paramo - https://github.com/JesusUrias
