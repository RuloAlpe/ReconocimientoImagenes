#cambios para ver si estan en el servidor.
#dfgsgdfdfhhdg

import cv2
import os
import numpy as np
from PIL import Image

# Para deteccion facial usamos Haar Cascade.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

lista = [".normal", ".lentes", ".risa", ".gorra", ".luzatras", ".luzderecha", ".luzizquierda" ,".sinluz", ".sorpresa", ".voltearder", ".voltearizq", ".abuelo"]
num = 0

# Para reconocimiento facial usamos LBPHFaceRecognizer
reconocimiento = cv2.createLBPHFaceRecognizer()

while num <= 11:

	def leer_imagenes_etiquetas(path):
		# Anadimos todas las imagenes en una lista image_paths
		# No se leen las imagenes con la extension .sad en el entrenemiento.
		image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith(lista[num])]
		# images contiene iimagenes de rostros
		images = []
		# labels contine las etiquetas que son asignadas para cada imagen.
		labels = []
		for image_path in image_paths:
			# Lee la imagen y la conviente a escala de grises.
			image_pil = Image.open(image_path).convert('L')
			# Convierte la imagen en formato numpy array.
			image = np.array(image_pil, 'uint8')
			# Obtiene la etiqueta de la imagen.
			nbr = int(os.path.split(image_path)[1].split(".")[0].replace("sujeto", ""))
			# Detecta el fostro en la imagen.
			faces = faceCascade.detectMultiScale(image)
			# Si el rostro es detectado anade el rostro a images y la etiqueta en labels.
			for (x, y, w, h) in faces:
				images.append(image[y: y + h, x: x + w])
				labels.append(nbr)
				cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
				cv2.waitKey(50)
		# Regresa la lista images y la lista labels
		return images, labels

	# Path de la carpeta que contiene los rostros
	path = './carpetaRaul'

	# Llama a la funcion get_images_and_labels y obtiene los rostros de images y su etiqueta en labels
	images, labels = leer_imagenes_etiquetas(path)
	cv2.destroyAllWindows()

	# Entrenamiento
	reconocimiento.train(images, np.array(labels))

	# Anadimos las imagenes con la extension .sad dentro de image_path.
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(lista[num])]
	for image_path in image_paths:
		predict_image_pil = Image.open(image_path).convert('L')
		predict_image = np.array(predict_image_pil, 'uint8')
		faces = faceCascade.detectMultiScale(predict_image)
		for (x, y, w, h) in faces:
			nbr_predicted, conf = reconocimiento.predict(predict_image[y: y + h, x: x + w])
			nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("sujeto", ""))
			if nbr_actual == nbr_predicted:
				print "La imagen {} correctamente reconocida con confianza de {}".format(lista[num],conf)
			else:
				print "La imagen {} incorrectamente reconocida".format(lista[num])
			cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
			cv2.waitKey(1000)


	num = num + 1		
