import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import math

class Drone3D:
    def __init__(self, x, y, z):
        self.pos = np.array([float(x), float(y), float(z)])
        self.pitch = 0  
        self.yaw = 0    
        self.roll = 0   
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])

        self.points = np.array([
            [-50, -15, -20], [50, -15, -20], [50, 15, -20], [-50, 15, -20], 
            [-50, -15, 20], [50, -15, 20], [50, 15, 20], [-50, 15, 20],
            [0, 0, -50] 
        ])
        
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (0, 8), (1, 8), (2, 8), (3, 8)
        ]

    def update(self, control_signals):
        self.yaw += control_signals['yaw'] * 0.02
        
        self.pitch += control_signals['pitch'] * 0.02

        forward_vec = np.array([
            math.cos(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            math.sin(self.yaw) * math.cos(self.pitch)
        ])
        self.acceleration = forward_vec * control_signals['forward'] * 0.1
        
        self.acceleration[1] -= control_signals['lift'] * 0.1

        self.velocity += self.acceleration
        self.pos += self.velocity
        self.velocity *= 0.95 

    def project(self, screen_width, screen_height, fov, viewer_distance):
        rotation_y = np.array([
            [math.cos(self.yaw), 0, math.sin(self.yaw)],
            [0, 1, 0],
            [-math.sin(self.yaw), 0, math.cos(self.yaw)]
        ])
        rotation_x = np.array([
            [1, 0, 0],
            [0, math.cos(self.pitch), -math.sin(self.pitch)],
            [0, math.sin(self.pitch), math.cos(self.pitch)]
        ])
        
        rotated_points = np.dot(self.points, rotation_x)
        rotated_points = np.dot(rotated_points, rotation_y)
        
        projected_points = []
        for point in rotated_points:
            z = point[2] + viewer_distance
            if z == 0: z = 0.01
            
            f = fov / z
            x = point[0] * f + screen_width / 2
            y = -point[1] * f + screen_height / 2
            projected_points.append((x, y))
        return projected_points

def main():
    pygame.init()
    
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Drone Simulation")
    font = pygame.font.SysFont("Arial", 18)

    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    drone = Drone3D(0, 0, 500)
    clock = pygame.time.Clock()

    control_signals = {'pitch': 0, 'yaw': 0, 'forward': 0, 'lift': 0}

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        img_h, img_w, _ = image.shape
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_3d = []
                face_2d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                head_x = angles[0]
                head_y = angles[1]

                control_signals['yaw'] = -head_y
                control_signals['pitch'] = head_x - 0.2 
                
                mouth_top = face_landmarks.landmark[13]
                mouth_bottom = face_landmarks.landmark[14]
                mouth_opening = abs(mouth_bottom.y - mouth_top.y)
                
                if mouth_opening > 0.04:
                    control_signals['lift'] = 1
                    control_signals['forward'] = 0
                else:
                    control_signals['lift'] = 0
                    control_signals['forward'] = 1
        else:
            control_signals = {'pitch': 0, 'yaw': 0, 'forward': 0, 'lift': 0}

        drone.update(control_signals)

        screen.fill((20, 20, 40))
        projected_points = drone.project(screen_width, screen_height, 256, 5)
        
        for edge in drone.edges:
            pygame.draw.line(screen, (255, 255, 255), projected_points[edge[0]], projected_points[edge[1]], 2)

        hud_text = font.render(f"Yaw: {control_signals['yaw']:.2f} Pitch: {control_signals['pitch']:.2f} Lift: {control_signals['lift']}", True, (200, 200, 200))
        screen.blit(hud_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()