
import numpy as np
from numpy import pi
import time

from gym import core, spaces
from gym.utils import seeding


class Acrobot(core.Env):
    """
    Modified version of the acrobot env:
    - continuous action instead of discrete
    - gravity is variable
    - both joints can be controlled

    https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    SPARSE_REWARDS = False
    first_distance = None

    dt = .1

    GRAVITY = 0.
    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = .1  #: [kg] mass of link 1
    LINK_MASS_2 = .1  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = .3 * np.pi
    MAX_VEL_2 = .3 * np.pi

    AVAIL_TORQUE = [-1., 0., +1]

    torque_noise_max = 0.

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    ts = None

    def __init__(self, max_ts):
        self.max_ts = max_ts
        self.viewer = None

        high = np.array([5., 5., 3., 3.,self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        high = np.array([1, 1])
        low = -high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-np.pi, high=np.pi, size=(4,))
        self.state[1] = 0
        self.state[2] = 0

        self.ts = 0

        while True:
            self.target_x = np.random.uniform(-2., 2.)
            self.target_y = np.random.uniform(-2., 2.)
            r = np.sqrt(np.power(self.target_x, 2) + np.power(self.target_y, 2))
            if r < 2.05:
                break

        self.first_distance = self._get_distance()

        return self._get_ob()

    def step(self, a):
        s = self.state

        # Add noise to the force action
        if self.torque_noise_max > 0:
            a[0] += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)
            a[1] += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        s_augmented = np.append(s, a[0])
        s_augmented = np.append(s_augmented, a[1])

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        ns = ns[-1]
        ns = ns[:4]

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)

        self.state = ns

        terminal = self._get_terminal()

        self.ts += 1

        if self.SPARSE_REWARDS:
            reward = -1 if not terminal else 10
        else:
            reward = self._get_reward() if not terminal else 10

        if self.ts > self.max_ts:
            terminal = True

        if 1.02*self.first_distance < self._get_distance():
            terminal = True

        print(reward)

        return (self._get_ob(), reward, terminal, {})

    def _get_reward(self):
        return -self._get_distance()

    def _get_distance(self):
        x, y = self._get_end_effector_coordinates()
        return np.linalg.norm(np.array((x, y)) - np.array((self.target_x, self.target_y)))

    def _get_ob(self):
        s = self.state
        x, y = self._get_end_effector_coordinates()

        dx = self.target_x - x
        dy = self.target_y - y

        return np.array([dx, dy, s[0], s[1], s[2], s[3]])

    def _get_angle_goal(self):
        x, y = self.target_x, self.target_y
        theta_rad = np.arctan2(x, -y)
        if theta_rad < 0.:
            theta_rad = theta_rad + 2 * np.pi
        return theta_rad

    def _get_angle_end_effector(self):
        """
        validated
        :return:
        """
        s = self.state
        theta_rad = s[0]  # radians
        if theta_rad < 0.:
            theta_rad = theta_rad + 2 * np.pi
        # theta_deg = theta_rad * 180./ np.pi
        return theta_rad

    def _get_end_effector_coordinates(self):
        s = self.state
        theta_1 = s[0]
        r_1 = self.LINK_LENGTH_1
        x_1 = + r_1 * np.sin(theta_1)
        y_1 = - r_1 * np.cos(theta_1)

        theta_2 = s[1] + s[0]
        r_2 = self.LINK_LENGTH_2
        x_2 = + r_2 * np.sin(theta_2)
        y_2 = - r_2 * np.cos(theta_2)

        return x_1 + x_2, y_1 + y_2

    def _get_angle_diff(self, angle_1, angle_2):
        """
        get shortest dist in angle space
        :param angle_1:
        :param angle_2:
        :return:
        """
        foo = abs(angle_1 - angle_2) + np.pi
        foo = foo % (2 * np.pi)
        foo = foo - np.pi
        if angle_1 > angle_2:
            return -foo
        return foo

    def _get_terminal(self):

        x, y = self._get_end_effector_coordinates()

        dist = np.sqrt(np.power(x - self.target_x, 2) + np.power(y - self.target_y, 2))

        if dist < 0.15:
            return True

        return False

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI

        g = self.GRAVITY
        a_1 = s_augmented[-2]
        a_2 = s_augmented[-1]
        s = s_augmented[:-1]

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        d1 = m1 * lc1 ** 2 + m2 * \
             (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) \
               + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a_2 + d2 / d1 * phi1 - phi2) / \
                       (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a_2 + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                       / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(a_1 + d2 * ddtheta2 + phi1) / d1

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0., 0.)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        time.sleep(.01)

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + .2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        # self.viewer.draw_line((-2.2, 1.8), (2.2, 1.8))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        # draw target
        target = self.viewer.draw_circle(radius=0.1)
        target.set_color(1., .2, .2)
        target.add_attr(rendering.Transform(translation=(self.target_x, self.target_y)))

        # draw end effector
        end_effector = self.viewer.draw_circle(radius=0.1)
        end_effector.set_color(.2, 1., .2)
        x, y = self._get_end_effector_coordinates()
        end_effector.add_attr(rendering.Transform(translation=(x, y)))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
