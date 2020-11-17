import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.
    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    x, y = list(zip(*path))
    x_displacement = np.array(x[1:]) - np.array(x[:-1])
    y_displacement = np.array(y[1:]) - np.array(y[:-1])

    times = np.zeros(len(x_displacement) + 1)
    for idx in range(1, len(x_displacement) + 1):
        times[idx] = times[idx-1] + np.sqrt(x_displacement[idx-1]**2 + y_displacement[idx-1]**2) / V_des
    
    x_tck = scipy.interpolate.splrep(times, x, s=alpha)
    y_tck = scipy.interpolate.splrep(times, y, s=alpha)
    t_smoothed = np.arange(0, times[-1], dt)
    
    traj_smoothed_x = scipy.interpolate.splev(t_smoothed, x_tck)
    traj_smoothed_y = scipy.interpolate.splev(t_smoothed, y_tck)
    traj_smoothed_xdot = scipy.interpolate.splev(t_smoothed, x_tck, der=1)
    traj_smoothed_ydot = scipy.interpolate.splev(t_smoothed, y_tck, der=1)
    traj_smoothed_xddot = scipy.interpolate.splev(t_smoothed, x_tck, der=2)
    traj_smoothed_yddot = scipy.interpolate.splev(t_smoothed, y_tck, der=2)
    traj_smoothed_theta = np.arctan2(traj_smoothed_ydot, traj_smoothed_xdot)
    traj_smoothed = np.vstack([traj_smoothed_x, 
                               traj_smoothed_y,
                              traj_smoothed_theta,
                              traj_smoothed_xdot,
                              traj_smoothed_ydot,
                              traj_smoothed_xddot,
                              traj_smoothed_yddot]).T
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
