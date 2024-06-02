from PIL import Image
import matplotlib.pyplot as plt
import math

# --- Parte 1: Mostrar el laberinto ---

# Cargamos la imagen del laberinto
laberinto_imagen = Image.open("C:/Users/osrpa/OneDrive/Escritorio/Documentos/2024/2024-1/inteligencia artificial/Lab IA/reconocimiento-coppelia (2)/reconocimiento-coppelia (1)/reconocimiento-coppelia/Cw.png")
laberinto_imagen = laberinto_imagen.convert("RGB")  # Convertimos la imagen a modo RGB

# Obtenemos las dimensiones de la imagen
ancho, alto = laberinto_imagen.size

# Creamos una lista para almacenar el mapeo del laberinto
laberinto = []

# Definir nuevas coordenadas de entrada y salida
entrada = (30, 170)  # Por ejemplo, en la posición (30, 170)
salida = (ancho - 35, alto - 25)  # Por ejemplo, en la posición (ancho - 35, alto - 25)

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

# --- Parte 2: Algoritmo A* ---

show_animation = True

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # Definir entrada y salida en el laberinto
    sx, sy = entrada
    gx, gy = salida

    # Convertir coordenadas a unidades de la rejilla
    grid_size = 1.0  # Tamaño de la rejilla (1 unidad por celda)
    robot_radius = 0.5  # Radio del robot

    # Crear listas de posiciones de obstáculos (ox, oy)
    ox, oy = [], []
    for y in range(alto):
        for x in range(ancho):
            if laberinto[y][x] == 1:
                ox.append(x)
                oy.append(-y)  # Invertir y para que coincida con la visualización

    if show_animation:  # pragma: no cover
        plt.figure()
        plt.plot(ox, oy, ".k")
        plt.plot(sx, -sy, "og")
        plt.plot(gx, -gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, [-y for y in ry], "-r")  # Invertir y para la visualización correcta
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
