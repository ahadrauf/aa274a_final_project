import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
import turtlebot_model as tb

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx,:2].mean(axis=0)
        th = self.xs[idx,2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # TODO: Update self.xs.
        # Hint: Call self.transition_model().
        # Hint: You may find np.random.multivariate_normal useful.

        # Generate noisy controls
        us = u + np.random.multivariate_normal(mean=np.zeros(2), cov=self.R, size=self.M)

        # Run forward transition
        self.xs = self.transition_model(us, dt)

        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """

        ########## Code starts here ##########
        # TODO: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.

        # Get sum of weights and cumsum array
        w_cumsum = np.cumsum(ws)
        w_cumsum /= w_cumsum[-1]

        # Sample randomly
        r = np.random.rand() / self.M

        # Get equally spaced points
        samples = np.linspace(0, 1.0, self.M, endpoint=False) + r

        # Sample indexes
        sample_idx = np.searchsorted(w_cumsum, samples)

        # Update xs from samples
        self.xs = xs[sample_idx]
        self.ws = ws[sample_idx]

        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns -- shape (2, J)
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # TODO: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: A simple solution can be using a for loop for each particle
        #       and a call to tb.compute_dynamics
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.
        # Hint: This faster/better solution does not use loop and does 
        #       not call tb.compute_dynamics. You need to compute the idxs
        #       where abs(om) > EPSILON_OMEGA and the other idxs, then do separate 
        #       updates for them

        # Grab relevant variables
        x, y, th = self.xs[:,0], self.xs[:,1], self.xs[:,2]
        V, om = us[:,0], us[:,1]

        # Get indexes where om is less than EPSILON_OMEGA
        small_om_idx = np.argwhere(abs(om) < EPSILON_OMEGA).squeeze(axis=-1)

        # Pre-compute some values
        s, c, ss, cc = np.sin(th), np.cos(th), np.sin(th + om * dt), np.cos(th + om * dt)

        # Compute values for small om
        ts = (dt / 2.) * (ss + s)
        tc = (dt / 2.) * (cc + c)

        # Compute g
        g_small_om = np.array([
            x + V * tc,
            y + V * ts,
            th + om * dt,
        ]).T

        # Compute values for normal om
        ts = ss - s
        tc = cc - c

        # Compute g
        g = np.array([
            x + V * ts / om,
            y - V * tc / om,
            th + om * dt,
        ]).T

        # Replace values in g at approriate indexes with small om ones where appropriate
        g[small_om_idx] = g_small_om[small_om_idx]

        ########## Code ends here ##########

        return g

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        ws = np.zeros_like(self.ws)

        ########## Code starts here ##########
        # TODO: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        # Hint: You'll need to call self.measurement_model()

        # Get joint innovation and covariances from measurements
        vs, Q = self.measurement_model(z_raw, Q_raw)    # shapes (M, 2I), (2I, 2I)

        # Get pdf likelihoods
        ws = scipy.stats.multivariate_normal.pdf(x=vs, mean=None, cov=Q)    # shape (M)

        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I]  - joint innovation for all measurements.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.array(Q_raw))

        ########## Code starts here ##########
        # TODO: Compute Q.
        # Hint: You might find scipy.linalg.block_diag() useful
        Q = scipy.linalg.block_diag(*Q_raw)

        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        ########## Code starts here ##########
        # TODO: Compute vs (with shape [M x I x 2]).
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       observed line, find the most likely map entry (the entry with 
        #       least Mahalanobis distance).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!
        # Hint: For the faster solution, you might find np.expand_dims(), 
        #       np.linalg.solve(), np.meshgrid() useful.

        # Compute predicted lines and get dimensions
        _, I = z_raw.shape
        hs = np.stack([self.compute_predicted_measurements()] * I, axis=-1) # shape (M, 2, J) --> (M, 2, J, I)
        M, _, J, _ = hs.shape

        # Compute innovations
        z = np.tile(np.expand_dims(np.expand_dims(z_raw, axis=1), axis=0), (M, 1, J, 1))    # shape (2, I) --> (M, 2, J, I)
        v_alpha = angle_diff(z[:,0,:,:], hs[:,0,:,:])   # shape (M, J, I)
        v_rho = z[:,1,:,:] - hs[:,1,:,:]                # shape (M, J, I)
        v = np.expand_dims(np.stack([v_alpha, v_rho], axis=-1), axis=-1)     # Shape (M, J, I, 2) --> (M, J, I, 2, 1)

        # Get Q inverses
        Q_inv = np.linalg.inv(Q_raw)     # shape (I, 2, 2)

        # Compute distances
        ds = np.matmul(v.transpose(0,1,2,4,3), np.matmul(Q_inv, v)).squeeze(axis=(-2, -1)).transpose(0, 2, 1) # Shape (M, J, I, 1, 1) --> (M, J, I) --> (M, I, J)

        # Find min over all J map elements in ds
        min_idx = np.argmin(ds, axis=2).flatten()     # Shape (M, I) --> Shape (M * I)
        v = v.squeeze(axis=-1).transpose(0, 2, 1, 3).reshape(-1, J, 2)       # shape (M, J, I, 2, 1) --> (M, J, I, 2) --> (M, I, J, 2) --> (M * I, J, 2)
        vs = v[np.arange(M * I), min_idx, :].reshape(M, I, 2)           # shape (M * I, 2) --> (M, I, 2)

        ########## Code ends here ##########

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,2,J] - J line parameters in the scanner (camera) frame for M particles.
        """
        ########## Code starts here ##########
        # TODO: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       map line, transform to scanner frmae using tb.transform_line_to_scanner_frame()
        #       and tb.normalize_line_parameters()
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebot_model functions directly here. This
        #       results in a ~10x speedup.
        # Hint: For the faster solution, it does not call tb.transform_line_to_scanner_frame()
        #       or tb.normalize_line_parameters(), but reimplement these steps vectorized.

        # Define vars
        J, M = self.map_lines.shape[1], self.xs.shape[0]
        alpha, r = np.stack([self.map_lines] * M, axis=1)                          # each is shape (M, J)
        x, y, th = np.stack([self.xs.T] * J, axis=2)                               # each is shape (M, J)

        # Define homogeneous rotation matrix to get camera pose to world frame
        x_cam, y_cam, th_cam = self.tf_base_to_camera  # Save the camera angle since this will be overwritten
        cam_pose = np.array([self.tf_base_to_camera[0], self.tf_base_to_camera[1], 1.])         # shape (3,)
        c, s = np.cos(th), np.sin(th)           # Each is shape (M, J)
        ca, sa = np.cos(alpha), np.sin(alpha)   # Each is shape (M, J)
        R = np.transpose(np.array([
            [c, -s, x],
            [s, c, y],
            [np.zeros((M, J)), np.zeros((M, J)), np.zeros((M, J))]
        ]), (2,3,0,1))       # shape (3, 3, M, J)  --> (M, J, 3, 3)
        cam_pose_world = np.matmul(R, cam_pose)     # shape (M, J, 3)
        # Replace the last element with cam angle in world frame
        cam_pose_world[:, :, 2] = th_cam + th

        # Calculate h = [alpha, r]
        hs = np.stack([
            alpha - th - th_cam,
            r - np.sum(cam_pose_world[:, :, :2] * np.stack([ca, sa], axis=2), axis=2),
        ], axis=1)              # shape (M, 2, J)

        # Normalize values
        neg_r = np.where(hs[:,1,:] < 0)
        hs[neg_r[0],0,neg_r[1]] += np.pi
        hs[neg_r[0],1,neg_r[1]] *= -1
        hs[:,0,:] = (hs[:,0,:] + np.pi) % (2 * np.pi) - np.pi

        ########## Code ends here ##########

        return hs

