from PIL import Image
import matplotlib.pyplot as plt

# Cargamos la imagen del laberinto
laberinto_imagen = Image.open("C:/Users/osrpa/OneDrive/Escritorio/Documentos/2024/2024-1/inteligencia artificial/Lab IA/reconocimiento-coppelia (2)/reconocimiento-coppelia (1)/reconocimiento-coppelia/Cw.png")
laberinto_imagen = laberinto_imagen.convert("RGB")  # Convertimos la imagen a modo RGB

# Obtenemos las dimensiones de la imagen
ancho, alto = laberinto_imagen.size

# Creamos una lista para almacenar el mapeo del laberinto
laberinto = []

# Definir nuevas coordenadas de entrada y salida
entrada = (30, 170)  # Por ejemplo, en la posición (5, 10)
salida = (ancho - 35, alto - 25)  # Por ejemplo, en la posición (ancho - 10, alto - 5)

# Convertimos la imagen en una matriz de píxeles y mapeamos las paredes
for y in range(alto):
    fila = []
    for x in range(ancho):
        # Obtenemos el color del píxel en la posición (x, y)
        color_pixel = laberinto_imagen.getpixel((x, y))
        # Si el píxel es negro, asumimos que es una pared
        if color_pixel == (0, 0, 0):
            fila.append(1)  # 1 representa una pared
        else:
            fila.append(0)  # 0 representa un espacio libre
    laberinto.append(fila)

# Creamos una figura
plt.figure()

# Dibujamos la matriz del laberinto
for y in range(alto):
    for x in range(ancho):
        if laberinto[y][x] == 1:
            plt.plot(x, -y, 'ks')  # Representamos una pared como un punto negro
        else:
            plt.plot(x, -y, 'ws')  # Representamos un espacio libre como un punto blanco

# Marcamos la entrada y la salida
plt.plot(entrada[0], -entrada[1], 'ro', label='Entrada')  # Marcamos la entrada en rojo
plt.plot(salida[0], -salida[1], 'go', label='Salida')    # Marcamos la salida en verde

# Configuramos el aspecto del gráfico
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Laberinto')

# Mostramos la leyenda
plt.legend()

# Mostramos el gráfico
plt.grid(True)
plt.show()
