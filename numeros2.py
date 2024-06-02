import math
import matplotlib.pyplot as plt
from PIL import Image

show_animation = True


class AStarPlanner:

    def __init__(self, laberinto, resolution, rr):
        self.resolution = resolution
        self.rr = rr
        self.obstacle_map = laberinto
        self.x_width = len(laberinto[0])
        self.y_width = len(laberinto)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(sx, sy, 0.0, -1)
        goal_node = self.Node(gx, gy, 0.0, -1)

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

            if show_animation:
                plt.plot(current.x, current.y, "xr", markersize=15)  # Red 'x' for current node
                plt.pause(0.0001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.get_motion_model()):
                node = self.Node(current.x + self.get_motion_model()[i][0],
                                 current.y + self.get_motion_model()[i][1],
                                 current.cost + self.get_motion_model()[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_index(self, node):
        return node.y * self.x_width + node.x

    def verify_node(self, node):
        if node.x < 0 or node.x >= self.x_width:
            return False
        elif node.y < 0 or node.y >= self.y_width:
            return False
        return not self.obstacle_map[node.y][node.x]

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

#C:/Users/osrpa/OneDrive/Escritorio/Documentos/2024/2024-1/inteligencia artificial/Lab IA/reconocimiento-coppelia (2)/reconocimiento-coppelia (1)/reconocimiento-coppelia
def main():
    laberinto_imagen = Image.open("C:/Users/osrpa/OneDrive/Escritorio/Documentos/2024/2024-1/inteligencia artificial/Lab IA/reconocimiento-coppelia (2)/reconocimiento-coppelia (1)/reconocimiento-coppelia/Cw.png")
    laberinto_imagen = laberinto_imagen.convert("RGB")
    ancho, alto = laberinto_imagen.size

    laberinto = []
    for y in range(alto):
        fila = []
        for x in range(ancho):
            color_pixel = laberinto_imagen.getpixel((x, y))
            if color_pixel == (0, 0, 0):
                fila.append(True)  # True representa una pared
            else:
                fila.append(False)  # False representa un espacio libre
        laberinto.append(fila)

    sx = 30  # Start x position
    sy = 170  # Start y position
    gx = ancho - 35  # Goal x position
    gy = alto - 25  # Goal y position
    grid_size = 5.0  # grid resolution [m]
    robot_radius = 0.1  # robot radius[m]

    if show_animation:
        plt.imshow(laberinto, cmap='Greys', origin='lower')

    a_star = AStarPlanner(laberinto, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

   


if __name__ == '__main__':
    main()
