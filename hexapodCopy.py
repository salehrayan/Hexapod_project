from core import HexapodCore
from time import sleep
import time
import numpy as np
import mujoco
import warnings


class Hexapod(HexapodCore):
    def __init__(self, mj_model, mj_data, viewer):
        super().__init__()
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.viewer = viewer

        self.CoxaLength = 0.09420844915929781 * 100
        self.FemurLength = 0.10000001058429393 * 100
        self.TibiaLength = 0.1435408051706695 * 100
        self.CoxaXLength = 0.07631

        self.leg_names = ['lm', 'lb', 'rb', 'rm', 'rf', 'lf']

        self.coxa_joint_points_original = np.array([[0.0, 0.1, 0.1435],
                                                    [-0.086603, 0.050001, 0.1435],
                                                    [-0.086604, -0.049999, 0.1435],
                                                    [0.0, -0.1, 0.1435],
                                                    [0.086601, -0.050001, 0.1435],
                                                    [0.086602, 0.049999, 0.1435]])
        xy_norm = np.linalg.norm(self.coxa_joint_points_original[:, :2], axis=1).reshape(-1, 1)
        self.coxa_joint_points = (self.coxa_joint_points_original[:, :2] +
                                  (self.coxa_joint_points_original[:, :2] * (self.CoxaXLength / xy_norm)))
        self.coxa_joint_points = np.concatenate((self.coxa_joint_points, np.ones((6, 1)) * 0.1435), axis=1)

        self.xy_origin = np.mean(self.coxa_joint_points, axis=0)

    def IK_LH_single(self, L, H, interval=0.1):
        # new_L = L - self.CoxaXLength
        A1 = np.arctan2(L, H)
        S_W = np.linalg.norm((L, H))
        if S_W > (self.FemurLength + self.TibiaLength):
            warnings.warn('S_W > femur+tibia, clipped', UserWarning)
        S_W = min(S_W, self.FemurLength + self.TibiaLength)
        A2 = np.arccos((self.FemurLength ** 2 + S_W ** 2 - self.TibiaLength ** 2) / (2 * S_W * self.FemurLength))
        femur_angle = A1 + A2
        femur_angle = np.rad2deg(A1 + A2)
        femur_angle = np.deg2rad(-femur_angle + 90)
        tibia_angle = np.arccos(
            (self.FemurLength ** 2 + self.TibiaLength ** 2 - S_W ** 2) / (2 * self.FemurLength * self.TibiaLength))
        tibia_angle = np.rad2deg(tibia_angle)
        tibia_angle = np.deg2rad(90 - tibia_angle)

        # self.mj_data.ctrl[[1, 2]] = [femur_angle, tibia_angle]
        # self.timer(interval+100)
        return femur_angle, tibia_angle

    def FK_LH_single(self, femur_angle, tibia_angle, interval=0.1):
        S_W = np.sqrt(self.FemurLength ** 2 + self.TibiaLength ** 2 - 2 * self.FemurLength * self.TibiaLength * np.cos(
            (np.pi / 2 - tibia_angle)))
        A1 = np.abs(femur_angle) + (np.pi / 2 - tibia_angle)
        A2 = np.arccos((self.TibiaLength ** 2 + S_W ** 2 - self.FemurLength ** 2) / (2 * self.TibiaLength * S_W + 1e-8))
        A3 = np.pi - (A1 + A2)

        L = S_W * np.cos(A3)
        # L = L_little + self.CoxaXLength
        H = S_W * np.sin(A3)
        return L, H

    def IK_translation_single(self, leg_num, tx=0, ty=0, tz=0, leg_idxs=[0, 1, 2], interval=0.5):
        x, y = self.coxa_joint_points[leg_num][:2] * 100
        coxa_angle, femur_angle, tibia_angle = 0, 0, 0
        L, H = self.FK_LH_single(femur_angle=femur_angle, tibia_angle=tibia_angle)

        coxa_axis_horizon_angle = np.arctan2(y, x)
        leg_horizon_angle = coxa_angle + coxa_axis_horizon_angle
        cos_leg_horizon_angle, sin_leg_horizon_angle = np.cos(leg_horizon_angle), np.sin(leg_horizon_angle)

        x_end = x + L * cos_leg_horizon_angle
        y_end = y + L * sin_leg_horizon_angle
        x_new = x + tx
        y_new = y + ty

        delta_x = x_end - x_new
        delta_y = y_end - y_new
        L_new = np.hypot(delta_x, delta_y)
        H_new = H + tz
        # print(delta_y)

        coxa_angle += self.arctan_shifted(delta_y, delta_x, np.arctan2(y, x)) - self.arctan_shifted(y, x,
                                                                                                    np.arctan2(y, x))
        femur_angle, tibia_angle = self.IK_LH_single(L_new, H_new)

        self.mj_data.ctrl[leg_idxs] = np.array([coxa_angle, femur_angle, tibia_angle])

    def IK_translation_all(self, tx=0, ty=0, tz=0, interval=100):
        for i in range(6):
            self.IK_translation_single(leg_num=i, tx=tx, ty=ty, tz=tz, leg_idxs=[i * 3, i * 3 + 1, i * 3 + 2])
        # self.timer(interval)

    def IK_rotation_translation_all_generator(self, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
        rx, ry, rz = -np.deg2rad(rx), -np.deg2rad(ry), np.deg2rad(rz)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx
        R_inv = R.T

        L, H = self.FK_LH_single(0, 0)
        translation_vectors = []
        final_vs = []
        final_vs_original = []
        end_points = []
        for i in range(6):
            # Translate point to the new origin, apply rotation, and translate back
            v = self.coxa_joint_points[i]
            original_v = self.coxa_joint_points_original[i]
            end_point = (v[:2] +
                         (v[:2] / np.linalg.norm(v[:2])) * L / 100)
            end_point = np.hstack((end_point, np.array([0])))

            translated_v = v - self.xy_origin
            translated_original_v = original_v - self.xy_origin

            rotated_v = R @ translated_v
            rotated_original_v = R @ translated_original_v
            rotated_end_point = R_inv @ end_point

            final_v = rotated_v + self.xy_origin + np.array([tx, ty, tz]) * 0.01
            final_v_original = rotated_original_v + self.xy_origin + np.array([tx, ty, tz]) * 0.01

            translation_vector = final_v - v
            translation_vectors.append(translation_vector)
            final_vs.append(final_v)
            final_vs_original.append(final_v_original)
            end_points.append(rotated_end_point)

        translation_vectors = np.array(translation_vectors) * 100
        final_vs = np.array(final_vs) * 100
        final_vs_original = np.array(final_vs_original) * 100
        end_points = np.array(end_points) * 100

        coxas, femurs, tibias = [], [], []

        for i in range(6):
            end_point = end_points[i]

            L_new = np.linalg.norm(end_point[:2] - final_vs[i, :2])
            H_new = H + translation_vectors[i, 2]
            femur_angle, tibia_angle = self.IK_LH_single(L=L_new, H=H_new)
            coxa_angle = -(self.arctan_shifted(final_vs_original[i, 1] - ty, final_vs_original[i, 0] - tx,
                                               np.arctan2(final_vs_original[i, 1] - ty, final_vs_original[i, 0] - tx)) -
                           self.arctan_shifted(end_point[1] - final_vs_original[i, 1],
                                               end_point[0] - final_vs_original[i, 0],
                                               np.arctan2(final_vs_original[i, 1] - ty, final_vs_original[i, 0] - tx))
                           )
            coxas.append(coxa_angle)
            femurs.append(femur_angle)
            tibias.append(tibia_angle)
        return np.array(coxas), np.array(femurs), np.array(tibias)

    def IK_rotation_translation_all(self, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
        coxas, femurs, tibias = self.IK_rotation_translation_all_generator(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz)

        self.mj_data.ctrl[0::3] = coxas
        self.mj_data.ctrl[1::3] = femurs
        self.mj_data.ctrl[2::3] = tibias

    def walk(self, repetitions=5, stride_x=5, strid_y=0, num_samples=10, interval=0.1):
        stride_norm = np.hypot(stride_x, strid_y)

        start_coxas, start_femurs, start_tibias = [], [], []

        tx_start = np.linspace(0, -stride_x / 2, num_samples, endpoint=False)
        ty_start = np.linspace(0, -strid_y / 2, num_samples, endpoint=False)

        for i in range(num_samples):
            coxa, femur, tibia = self.IK_rotation_translation_all_generator(tx=tx_start[i], ty=ty_start[i])
            start_coxas.append(coxa)
            start_femurs.append(femur)
            start_tibias.append(tibia)
        start_coxas, start_femurs, start_tibias = np.array(start_coxas), np.array(start_femurs), np.array(start_tibias)

        coxas, femurs, tibias = [], [], []

        tx = np.linspace(-stride_x / 2, stride_x / 2, 2 * num_samples, endpoint=False)
        ty = np.linspace(-strid_y / 2, strid_y / 2, 2 * num_samples, endpoint=False)

        for i in range(2 * num_samples):
            coxa, femur, tibia = self.IK_rotation_translation_all_generator(tx=tx[i], ty=ty[i])
            coxas.append(coxa)
            femurs.append(femur)
            tibias.append(tibia)
        coxas, femurs, tibias = np.array(coxas), np.array(femurs), np.array(tibias)

        # Start, prepare
        femur_half_circle = (-np.pi / 6) * np.sin(np.linspace(0, np.pi, num_samples))
        for i in range(num_samples):
            # Tripod 1
            self.mj_data.ctrl[3::6] = start_coxas[i, 1::2]
            self.mj_data.ctrl[4::6] = start_femurs[i, 1::2]
            self.mj_data.ctrl[5::6] = start_tibias[i, 1::2]

            # Tripod 2
            self.mj_data.ctrl[0::6] = -start_coxas[i, 0::2]
            self.mj_data.ctrl[1::6] = -start_femurs[i, 0::2] + femur_half_circle[i]
            self.mj_data.ctrl[2::6] = -start_tibias[i, 0::2]
            self.timer(interval)

        # Initiate walking sequence
        femur_half_circle_twice = (-np.pi / 6) * np.sin(np.linspace(0, np.pi, 2 * num_samples))
        tri1_coxas = np.vstack((coxas, -coxas))
        tri1_femurs = np.vstack((femurs, -femurs + femur_half_circle_twice[:, np.newaxis]))
        tri1_tibias = np.vstack((tibias, -tibias))

        tri2_coxas = np.vstack((-coxas, coxas))
        tri2_femurs = np.vstack((-femurs + femur_half_circle_twice[:, np.newaxis], femurs))
        tri2_tibias = np.vstack((-tibias, tibias))

        for r in range(repetitions):
            for i in range(4 * num_samples):
                # Tripod 1
                self.mj_data.ctrl[3::6] = tri1_coxas[i, 1::2]
                self.mj_data.ctrl[4::6] = tri1_femurs[i, 1::2]
                self.mj_data.ctrl[5::6] = tri1_tibias[i, 1::2]

                # Tripod 2
                self.mj_data.ctrl[0::6] = tri2_coxas[i, 0::2]
                self.mj_data.ctrl[1::6] = tri2_femurs[i, 0::2]
                self.mj_data.ctrl[2::6] = tri2_tibias[i, 0::2]
                self.timer(interval)

    def rotation_dance(self, repetitions=3, max_angle=30, interval=0.5, num_samples=50, renderer=None):
        phase = np.linspace(0, 2 * np.pi, num_samples)[:-1]
        ry_s = max_angle * np.cos(phase)
        rx_s = max_angle * np.sin(phase)
        frames = []

        coxas, femurs, tibias = [], [], []
        for i in range(max_angle):
            coxa, femur, tibia = self.IK_rotation_translation_all_generator(rx=0, ry=i, tz=-5)
            coxas.append(coxa)
            femurs.append(femur)
            tibias.append(tibia)

        for i in range(num_samples - 1):
            coxa, femur, tibia = self.IK_rotation_translation_all_generator(rx=rx_s[i], ry=ry_s[i], tz=-5)
            coxas.append(coxa)
            femurs.append(femur)
            tibias.append(tibia)

        for i in range(max_angle):
            self.mj_data.ctrl[0::3] = coxas[i]
            self.mj_data.ctrl[1::3] = femurs[i]
            self.mj_data.ctrl[2::3] = tibias[i]
            self.timer(t=interval)
            # renderer.update_scene(self.mj_data, camera='hexapod_camera')
            # pixels = renderer.render()
            # frames.append(pixels)

        for _ in range(repetitions):
            for i in range(num_samples - 1):
                self.mj_data.ctrl[0::3] = coxas[max_angle + i]
                self.mj_data.ctrl[1::3] = femurs[max_angle + i]
                self.mj_data.ctrl[2::3] = tibias[max_angle + i]
                self.timer(t=interval)
                # renderer.update_scene(self.mj_data, camera='hexapod_camera')
                # pixels = renderer.render()
                # frames.append(pixels)
        return frames

    def boot_up(self):

        self.look()
        self.lie_down()
        self.curl_up()
        self.lie_flat()
        self.get_up()

    def shut_down(self):

        self.look()
        self.lie_down()
        self.lie_flat()
        self.curl_up(die=True)

    def curl_up(self, die=False, t=0.2):

        for leg in self.legs:
            leg.pose(hip_angle=0,
                     knee_angle=-(leg.knee.max + leg.knee.leeway),
                     ankle_angle=leg.ankle.max)

        sleep(t)

        if die: self.off()

    def lie_flat(self, t=0.15):

        for leg in self.legs:
            leg.pose()

        sleep(t)

    def lie_down(self, maxx=50, step=4, t=0.15):

        for angle in range(maxx, -(maxx + 1), -step):
            self.squat(angle)

        sleep(t)

    def get_up(self, maxx=70, step=4):

        for angle in range(-maxx, maxx + 1, step):
            self.squat(angle)

        self.default()

    def look(self, angle=0, t=0.05):
        self.neck.pose(angle)
        sleep(t)

    def twist_hip(self, angle=0, t=0.1):

        for hip in self.hips:
            hip.pose(angle)

        sleep(t)

    def squat(self, angle, t=0):

        for leg in self.legs:
            leg.move(knee_angle=angle)

        sleep(t)

    def rotate(self, offset=40, raised=-30, floor=50, repetitions=5, t=0.2):
        """ if offset > 0, hexy rotates left, else right """

        for r in range(repetitions):
            # replant tripod2 with an offset
            self.uniform_move(self.tripod2, None, raised, t)
            self.uniform_move(self.tripod2, offset, floor, t)

            # raise tripod1
            self.uniform_move(self.tripod1, -offset, raised)

            # swing tripod2's hips to an -offset
            self.uniform_move(self.tripod2, -offset, None, t)

            # lower tripod1
            self.uniform_move(self.tripod1, 0, floor, t)

    def stride(self, first_tripod, second_tripod, swing, raised, floor, t):
        """ first_tripod's legs replant to propel towards a direction while
            second_tripod's legs retrack by swinging to the opposite direction """

        a = np.deg2rad(np.array(self.simultaneous_move(first_tripod, swings=swing, knee_angle=raised)))
        self.timer(t)

        # sleep(t)

        b = np.deg2rad(np.array(self.simultaneous_move(second_tripod, swing[::-1])))
        self.timer(t + 0.1)
        c = np.deg2rad(np.array(self.simultaneous_move(first_tripod, swing, floor)))
        self.timer(t)

        # sleep(t)
        return a, (b, c)

    def tilt_side(self, left_angle=50, right_angle=0, t=0.2):
        """ if left_angle > right_angle, left side is higher than right side """

        self.uniform_move(legs=self.left_legs, knee_angle=left_angle)
        self.uniform_move(legs=self.right_legs, knee_angle=right_angle)
        sleep(t)

    def tilt(self, front_angle=50, middle_angle=25, back_angle=0, t=0.2):
        """ if front_angle > middle_angle > back_angle hexy's front is higher than his back """

        self.right_front.move(knee_angle=front_angle)
        self.left_front.move(knee_angle=front_angle)

        self.right_middle.move(knee_angle=middle_angle)
        self.left_middle.move(knee_angle=middle_angle)

        self.right_back.move(knee_angle=back_angle)
        self.left_back.move(knee_angle=back_angle)

        sleep(t)

    def default(self, offset=45, floor=60, raised=-30, t=0.2):
        """ Hexy's default pose, offset > 0 brings the front and back legs to the side """

        swings = [offset, 0, -offset]

        self.look()
        self.squat(floor, t)

        self.simultaneous_move(self.tripod1, swings, raised, t)
        self.simultaneous_move(self.tripod1, swings, floor, t)
        self.simultaneous_move(self.tripod2, swings[::-1], raised, t)
        self.simultaneous_move(self.tripod2, swings[::-1], floor, t)

    def uniform_move(self, legs, hip_angle=None, knee_angle=None, t=0):
        """ moves all legs with hip_angle, knee_angle """

        for leg in legs:
            leg.move(knee_angle, hip_angle)

        sleep(t)

    def simultaneous_move(self, legs, swings=[None, None, None], knee_angle=None, t=0):
        """ moves all legs with knee_angle to the respective hip angles at 'swing' """
        tripod_legs = []
        for leg, hip_angle in zip(legs, swings):
            leg_angles = leg.move(knee_angle, hip_angle)
            self.mj_data.ctrl[leg.leg_idxs] = leg_angles
            if leg.name == 'left front':
                print(leg_angles)
            # mujoco.mj_step(self.mj_model, self.mj_data)
            tripod_legs.append(leg_angles)

        # sleep(t)
        return tripod_legs

    import numpy as np

    @staticmethod
    def arctan_shifted(y, x, transition_angle):
        """
        Computes the arctangent of y/x, with a custom transition point for the output angle.

        Parameters:
        y (float or array-like): y-coordinate or array of y-coordinates.
        x (float or array-like): x-coordinate or array of x-coordinates.
        transition_angle (float): Desired transition angle in radians.

        Returns:
        float or ndarray: The shifted arctan2 angle(s) in the range [-pi, pi].
        """
        # Calculate the angle using np.arctan2
        theta = np.arctan2(y, x)

        # Shift the angle by the transition point
        shifted_theta = theta - transition_angle

        # Normalize to the range -pi to pi
        shifted_theta = (shifted_theta + np.pi) % (2 * np.pi)

        return shifted_theta

    def timer(self, t):
        # Record the real time when the simulation starts
        real_start_time = time.time()
        sim_start_time = self.mj_data.time

        while (time.time() - real_start_time) < t:
            # Step the simulation
            mujoco.mj_step(self.mj_model, self.mj_data)

            # Calculate the desired simulation time based on real time
            elapsed_real_time = time.time() - real_start_time
            desired_sim_time = sim_start_time + elapsed_real_time

            # If simulation time has caught up with the real time, sync the viewer
            if self.mj_data.time >= desired_sim_time:
                self.viewer.sync()

