from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
import numpy as np
import detect_and_align as detectar_y_alinear
import cv2
import os



class IdPersona:
    """Keeps track of known identities and calculates id matches"""

    def __init__(
        self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distancia_umbral
    ):
        print("Cargando todas las personas conocidas: ", end="")
        self.distancia_umbral = distancia_umbral
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None

        carpeta_imagenes = []
        os.makedirs(id_folder, exist_ok=True)
        ids = os.listdir(os.path.expanduser(id_folder))
        if not ids:
            return

        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            carpeta_imagenes = carpeta_imagenes + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Encontré %d imagenes en total" % len(carpeta_imagenes))
        fotos_alineadas, id_image_paths = self.detect_id_faces(carpeta_imagenes)
        feed_dict = {images_placeholder: fotos_alineadas, phase_train_placeholder: False}
        
        #junta los embeddings de la carpeta de todas las fotos y del dataset descargado de internet
        #el primer parametro (embedding) es el tensor de caracteristicas del modelo preentrenado
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def detect_id_faces(self, carpeta_imagenes):
        fotos_alineadas = []
        id_image_paths = []
        for image_path in carpeta_imagenes:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detectar_y_alinear.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print(
                    "[ALERTA] Se reconocieron varias caras a la vez.: %s" % image_path
                    + "\nEs importante que solo esté una persona en la cámara  "
                    + "Si crees que es un falso negativo, puedes resolverlo incrementando el umbral de la red neuronal"
                )
            fotos_alineadas = fotos_alineadas + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(fotos_alineadas), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        if self.id_names:
            personas_reconocidas = []
            distancias_personas_reconocidas = []
            
            #embs son las caracteristicas del frame de la cámara
            #self.embeddings son las caracteristicas de la carpeta donde están todas las ftoso
            #Se saca la distancia de los embeddings del frame con el embeddings 
            
            #se compara la distancia minima entre todas las personas personas_reconocidas
            #Y es la que mas se acerca a las personas reconocidas
            distance_matrix = pairwise_distances(embs, self.embeddings)
            for distance_row in distance_matrix:

                min_index = np.argmin(distance_row)
                if distance_row[min_index] < self.distancia_umbral:
                    personas_reconocidas.append(self.id_names[min_index])
                    distancias_personas_reconocidas.append(distance_row[min_index])
                else:
                    personas_reconocidas.append(None)
                    distancias_personas_reconocidas.append(None)
        else:
            personas_reconocidas = [None] * len(embs)
            distancias_personas_reconocidas = [np.inf] * len(embs)
        return personas_reconocidas, distancias_personas_reconocidas

    """
    Se carga modelo ResNet V1 utilizando un conjunto de datos de entrenamiento MS-Celeb-1M
    20170512-110547 (1.0) y se utiliza para detectar caras en una imagen. 
    """
    
def cargar_modelo(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Cargando el modelo FACENET, llamado: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Error, esperado: nombre del modelo no la ruta!")

