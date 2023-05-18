import numpy as np

import scipy

from tqdm.autonotebook import tqdm


class _base:
    def __init__(self, tbin, track_samples, arena):
        r"""
        Arena object can be a wall or circle.
        """
        self.tbin = tbin
        self.track_samples = track_samples

        self.arena = []
        for a in arena:
            if a[0] == "arc":
                o = self.create_arc(*a[1])
            elif a[0] == "wall":
                o = self.create_wall(*a[1])
            else:
                raise NotImplementedError
            self.arena.append((a[0], o))

    def create_wall(self, x_0, x_1, tc):
        vdir = x_1 - x_0
        L = np.linalg.norm(vdir)
        vdir /= L
        ndir = np.array([-vdir[1], vdir[0]])
        ang = np.angle(vdir[0] + 1j * vdir[1])
        return x_0, ang, L, vdir, ndir, tc

    def collide_wall(self, pos, wall):
        r"""
        Specifies wall with normal facing to the right when x_0 is at bottom, x_1 at top.
        Thickness is only in the direction of the normal, length is unaffected.
        """
        x_0, ang, L, vdir, ndir, thickness = wall
        pos_ = pos - x_0
        L_h = np.dot(pos_, vdir)
        L_v = np.dot(pos_, ndir)
        ang = np.angle(ndir[0] + 1j * ndir[1]) % (2 * np.pi)
        r = L_v - np.sign(L_v) * thickness

        if L_h > 0 and L_h < L and np.abs(L_v) < thickness:
            pos[0] -= r * ndir[0]
            pos[1] -= r * ndir[1]
            collide = True
        else:
            collide = False

        return collide, r, ang, pos  # +ve is on the side of ndir

    def create_arc(self, x_0, ang_0, ang_1, radius, tc):
        return x_0, ang_0, ang_1, radius, tc

    def collide_arc(self, pos, arc):
        r"""
        Specifies part of circular wall for collisions.
        """
        x_0, ang_0, ang_1, radius, thickness = arc
        pos_ = pos - x_0
        ang = np.angle(pos_[0] + 1j * pos_[1]) % (2 * np.pi)
        L = np.linalg.norm(pos_)
        L_v = L - radius
        r = L_v - np.sign(L_v) * thickness  # distance to boundary

        if ang > ang_0 and ang < ang_1 and np.abs(L_v) < thickness:
            pos[0] -= r * np.cos(ang)
            pos[1] -= r * np.sin(ang)
            collide = True
        else:
            collide = False

        return collide, r, ang, pos  # +ve is outside

    def compute_theta(self, sim_samples, theta_period, theta_offset):
        theta_t = np.empty(sim_samples)  # radians
        theta_t.fill(2 * np.pi * self.tbin / theta_period)
        theta_t = np.cumsum(theta_t) + theta_offset
        return theta_t


class animal_Lever(_base):
    r"""
    Rayleigh distribution parameter b: 13-17 cm/sec
    Gaussian omega distribution: mu 0-\pm 2.5 deg/sec, sigma 330-350 deg/sec
    Based on the model in [1], but removed instantaneous head direction changes and
    less movement parallel to the wall

    References:

    [1] `Modeling Boundary Vector Cell Firing Given Optic Flow as a Cue`,
        Florian Raudies, Michael E. Hasselmo

    """

    def __init__(self, tbin, track_samples, arena):
        super().__init__(tbin, track_samples, arena)

    def sample(
        self,
        pos_ini,
        v_b,
        omega_mu,
        omega_std,
        theta_period,
        theta_offset,
        wall_alert,
        wall_turn,
        wall_exit_angle=0.0,
        hd_noise=0.0,
        switch_avg=0,
    ):
        r"""
        :param float v_b: the scale parameter of the Rayleigh distribution,
                          mean is :math:`\sqrt{\frac{\pi}{2}}v_b`.
        """
        sim_samples = self.track_samples

        rv_v = scipy.stats.rayleigh(scale=v_b).rvs(sim_samples)
        rv_hd = scipy.stats.norm(loc=omega_mu, scale=omega_std).rvs(sim_samples)

        if switch_avg > 0:  # exponentially distributed intervals of no head turn
            taus = np.cumsum(
                np.random.exponential(switch_avg, size=(4 * sim_samples // switch_avg,))
            ).astype(int)
            taus = taus[taus < sim_samples]
            for k in np.arange(1, len(taus) - 1, 2):
                rv_hd[taus[k] : taus[k + 1]] = 0.0

        x_t = np.empty(sim_samples)  # mm
        y_t = np.empty(sim_samples)
        hd_t = np.empty(sim_samples)  # radians

        dir_t = np.empty(sim_samples)
        s_t = rv_v  # mm/s

        x_t[0] = pos_ini[0]
        y_t[0] = pos_ini[1]
        dir_t[0] = np.random.rand(1) * 2 * np.pi

        iterator = tqdm(range(1, sim_samples))
        for t in iterator:
            x_t[t] = x_t[t - 1] + np.cos(dir_t[t - 1]) * rv_v[t - 1] * self.tbin
            y_t[t] = y_t[t - 1] + np.sin(dir_t[t - 1]) * rv_v[t - 1] * self.tbin
            dir_t[t] = dir_t[t - 1] + rv_hd[t - 1] * self.tbin

            for a in self.arena:  # single iteration over all objects
                if a[0] == "arc":
                    r = self.collide_arc(np.array([x_t[t], y_t[t]]), a[1])
                elif a[0] == "wall":
                    r = self.collide_wall(np.array([x_t[t], y_t[t]]), a[1])

                if np.abs(r[1]) < wall_alert:  # approaching a boundary

                    rv_v[t] *= np.abs(r[1]) / wall_alert  # slow down

                    if r[0] is True:
                        x_t[t] = r[3][0]
                        y_t[t] = r[3][1]

                    if r[1] > 0:
                        delta = (dir_t[t : t + 1] - r[2]) % (2 * np.pi)
                        delta[delta > np.pi] -= 2 * np.pi
                    elif r[1] < 0:
                        delta = (dir_t[t : t + 1] - r[2] - np.pi) % (2 * np.pi)
                        delta[delta > np.pi] -= 2 * np.pi

                    if delta < -wall_exit_angle:
                        dir_t[t] += wall_turn
                    elif delta > wall_exit_angle:
                        dir_t[t] -= wall_turn

                    break  # only handle one at a time

        hd_t = dir_t + np.random.randn(sim_samples) * hd_noise

        hd_t = hd_t % (2 * np.pi)
        dir_t = dir_t % (2 * np.pi)
        theta_t = self.compute_theta(sim_samples, theta_period, theta_offset)

        return sim_samples, x_t, y_t, s_t, dir_t, hd_t, theta_t


class animal_random_walk(_base):
    r"""
    Biased random walk model of animal trajectories (OU process)
    """

    def __init__(self, tbin, track_samples, arena):
        super().__init__(tbin, track_samples, arena)

    def sample(self, pos_ini, mu, std, alpha, theta_period, theta_offset):
        sim_samples = self.track_samples

        x_t = np.empty(sim_samples)  # mm
        y_t = np.empty(sim_samples)
        hd_t = np.empty(sim_samples)  # radians

        x_t[0] = pos_ini[0]
        y_t[0] = pos_ini[1]
        sigma_x = std[0] * np.sqrt(
            1 - alpha[0] ** 2
        )  # std(x) = sigma_x/sqrt(1-alpha^2)
        sigma_y = std[1] * np.sqrt(
            1 - alpha[1] ** 2
        )  # std(y) = sigma_y/sqrt(1-alpha^2)

        iterator = tqdm(range(1, sim_samples))
        for t in iterator:
            x_t[t] = (
                alpha[0] * (x_t[t - 1] - mu[0]) + sigma_x * np.random.randn() + mu[0]
            )
            y_t[t] = (
                alpha[1] * (y_t[t - 1] - mu[1]) + sigma_y * np.random.randn() + mu[1]
            )

            for a in self.arena:
                if a[0] == "arc":
                    r = self.collide_arc(np.array([x_t[t], y_t[t]]), a[1])
                elif a[0] == "wall":
                    r = self.collide_wall(np.array([x_t[t], y_t[t]]), a[1])

                if r[0] != 0:
                    x_t[t] = r[2][0]
                    y_t[t] = r[2][1]

        emp_s_t = np.empty(sim_samples)
        dir_t = np.empty(sim_samples)

        vx_t = np.empty(sim_samples)
        vy_t = np.empty(sim_samples)
        vx_t[1:-1] = (x_t[2:] - x_t[:-2]) / (self.tbin * 2)  # mm/s
        vy_t[1:-1] = (y_t[2:] - y_t[:-2]) / (self.tbin * 2)  # mm/s
        vx_t[0] = vx_t[1]  # copy ends
        vy_t[sim_samples - 1] = vy_t[sim_samples - 2]

        emp_s_t = np.sqrt(vx_t**2 + vy_t**2)
        dir_t = np.angle(vx_t + vy_t * 1j)
        hd_t = dir_t + np.random.rand(sim_samples) * 0.1

        hd_t = hd_t % (2 * np.pi)
        dir_t = dir_t % (2 * np.pi)
        theta_t = self.compute_theta(sim_samples, theta_period, theta_offset)

        return sim_samples, x_t, y_t, emp_s_t, dir_t, hd_t, theta_t


class animal_straight_lines(_base):
    r"""
    Straight line based walk model of animal trajectories
    """

    def __init__(self, tbin, track_samples, arena):
        super().__init__(tbin, track_samples, arena)

    def sample(self, disorder, pos_ini, v_mean, theta_period, theta_offset):
        sim_samples = self.track_samples

        x_t = np.empty(sim_samples)  # mm
        y_t = np.empty(sim_samples)
        hd_t = np.empty(sim_samples)  # radians

        dir_t = np.empty(sim_samples)
        s_t = np.empty(sim_samples)  # mm/s

        x_t[0] = pos_ini[0]
        y_t[0] = pos_ini[1]

        def rehead(tt):
            dir_t[tt] = np.random.rand() * 2 * np.pi
            s_t[tt] = np.random.exponential(v_mean)

        rehead(0)
        arr = np.random.rand(sim_samples - 1)

        iterator = tqdm(range(1, sim_samples))
        for t in iterator:
            if arr[t - 1] < disorder:
                dis = True
            else:
                dis = False

            x_t[t] = np.cos(dir_t[t - 1]) * s_t[t - 1] * self.tbin + x_t[t - 1]
            y_t[t] = np.sin(dir_t[t - 1]) * s_t[t - 1] * self.tbin + y_t[t - 1]
            dir_t[t] = dir_t[t - 1]
            s_t[t] = s_t[t - 1]

            if dis:
                rehead(t)

            for a in self.arena:
                if a[0] == "arc":
                    r = self.collide_arc(np.array([x_t[t], y_t[t]]), a[1])
                elif a[0] == "wall":
                    r = self.collide_wall(np.array([x_t[t], y_t[t]]), a[1])

                if r[0] != 0:
                    x_t[t] = r[3][0]
                    y_t[t] = r[3][1]
                    rehead(t)
                    break

        hd_t = dir_t + np.random.rand(sim_samples) * 0.1

        hd_t = hd_t % (2 * np.pi)
        dir_t = dir_t % (2 * np.pi)
        theta_t = self.compute_theta(sim_samples, theta_period, theta_offset)

        return sim_samples, x_t, y_t, s_t, dir_t, hd_t, theta_t


def get_arena(arena_type):
    if arena_type == "box":
        left_x = 0.0
        right_x = 500.0
        bottom_y = 0.0
        top_y = 500.0
        tc = 10.0
        arena_width = right_x - left_x
        arena_height = top_y - bottom_y

        arena = [
            (
                "wall",
                (
                    np.array([right_x + tc, bottom_y - tc]),
                    np.array([left_x - tc, bottom_y - tc]),
                    tc,
                ),
            ),  # bottom
            (
                "wall",
                (
                    np.array([left_x - tc, bottom_y - tc]),
                    np.array([left_x - tc, top_y + tc]),
                    tc,
                ),
            ),  # left
            (
                "wall",
                (
                    np.array([left_x - tc, top_y + tc]),
                    np.array([right_x + tc, top_y + tc]),
                    tc,
                ),
            ),  # top
            (
                "wall",
                (
                    np.array([right_x + tc, top_y + tc]),
                    np.array([right_x + tc, bottom_y - tc]),
                    tc,
                ),
            ),
        ]  # right

    elif arena_type == "circle":
        radius = 150.0
        tc = 10.0
        arena = [("arc", (np.array([radius, radius]), 0.0, 2 * np.pi, radius + tc, tc))]

    elif arena_type == "hexagon":
        lat = 200.0
        tc = 10.0

        arena = [
            ("wall", (np.array([lat, 0.0]), np.array([0.0, 0.0]), tc)),
            (
                "wall",
                (
                    np.array([-lat / 2.0, lat * np.sqrt(3) / 2.0]),
                    np.array([0.0, 0.0]),
                    tc,
                ),
            ),
            (
                "wall",
                (
                    np.array([0.0, lat * np.sqrt(3)]),
                    np.array([-lat / 2.0, lat * np.sqrt(3) / 2.0]),
                    tc,
                ),
            ),
            (
                "wall",
                (
                    np.array([lat, lat * np.sqrt(3)]),
                    np.array([0.0, lat * np.sqrt(3)]),
                    tc,
                ),
            ),
            (
                "wall",
                (
                    np.array([lat * 1.5, lat * np.sqrt(3) / 2.0]),
                    np.array([lat, lat * np.sqrt(3)]),
                    tc,
                ),
            ),
            (
                "wall",
                (
                    np.array([lat, 0.0]),
                    np.array([lat * 1.5, lat * np.sqrt(3) / 2.0]),
                    tc,
                ),
            ),
        ]

    else:
        raise ValueError

    return arena
