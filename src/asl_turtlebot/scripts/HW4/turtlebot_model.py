import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    # xdot(t) = V(cos(theta(t))
    # ydot(t) = Vcos(theta(t))
    # thetadot(t) = omega --> dtheta = omega*dt
    x0, y0, theta0 = xvec
    V, omega = u
    
    if np.abs(omega) < EPSILON_OMEGA:
        g = xvec + dt*np.array([V*np.cos(theta0),
                                V*np.sin(theta0),
                                omega]) # x(t) = g(x_{t-1}, u_{t})
        Gx = np.array([[1, 0, -dt*V*np.sin(theta0)],
                       [0, 1, dt*V*np.cos(theta0)],
                       [0, 0, 1]])
        
        # These limits were evaluated using Wolfram Alpha
        # E.g. https://www.wolframalpha.com/input/?i=limit+of+v*t*sin%28a%2Bwt%29%2Fw+-+v%2Fw%5E2*%28-cos%28a%2Bwt%29%2Bcos%28a%29%29+as+w+goes+to+zero
        Gu = np.array([[dt*np.cos(theta0), -0.5*dt*dt*V*np.sin(theta0)],
                       [dt*np.sin(theta0), 0.5*dt*dt*V*np.cos(theta0)],
                       [0, dt]])
    else:
        # int(xdot) = x-x0 = int(Vcos(theta)dt) = int(V/omega * cos(theta) dtheta) from theta0 to theta0+omega*dt
        #                  = V/omega * (np.sin(theta0+omega*dt) - np.sin(theta0))
        g = xvec + np.array([V/omega * (np.sin(theta0+omega*dt) - np.sin(theta0)),
                             V/omega * (-np.cos(theta0+omega*dt) + np.cos(theta0)),
                             dt*omega])
        Gx = np.array([[1, 0, V/omega*(np.cos(theta0+omega*dt) - np.cos(theta0))],
                       [0, 1, V/omega*(np.sin(theta0+omega*dt) - np.sin(theta0))],
                       [0, 0, 1]])
        Gu = np.array([[1./omega * (np.sin(theta0+omega*dt) - np.sin(theta0)), 
                        -V*(np.sin(theta0+omega*dt) - np.sin(theta0))/(omega**2) + V/omega*dt*np.cos(theta0+omega*dt)],
                       [1./omega * (-np.cos(theta0+omega*dt) + np.cos(theta0)), 
                        -V*(-np.cos(theta0+omega*dt) + np.cos(theta0))/(omega**2) + V/omega*dt*np.sin(theta0+omega*dt)],
                       [0, dt]])
    
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True, get_global_cam_pose=False):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
         get_global_cam_pose: bool      - if True, will also return the camera poase in the global frame
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    alpha, r = line
    x, y, theta = x
    x_rel, y_rel, theta_rel = tf_base_to_camera
    R_func = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    R_rel = R_func(theta)
    
    # Calculate pose of camera in world frame
    x_cam, y_cam = np.array([x, y]) + np.matmul(R_rel, np.array([x_rel, y_rel]))
    theta_cam = theta + theta_rel
    R_cam = R_func(theta_cam)
    cam_pose_world = np.array([x_cam, y_cam, theta_cam])
    
    # Convert notable points on the line to the camera frame
    # The world frame's line is defined by x*cos(alpha) + y*sin(alpha) = r
    # We can get the points (r/cos(alpha), 0) and (0, r/sin(alpha)) in the world frame
    r_cam = r - x_cam*np.cos(alpha) - y_cam*np.sin(alpha)
    alpha_cam = alpha - theta_cam
    h = np.array([alpha_cam, r_cam])
    
    # dalpha = -dtheta_cam
    # dr = -dx_cam*cos(alpha) - dy_cam*sin(alpha)
    dx_cam = np.array([1, 0, -x_rel*np.sin(theta) - y_rel*np.cos(theta)])
    dy_cam = np.array([0, 1, x_rel*np.cos(theta) - y_rel*np.sin(theta)])
    dr = -dx_cam*np.cos(alpha) - dy_cam*np.sin(alpha)
    Hx = np.array([[0, 0, -1], dr])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h if not get_global_cam_pose else h, cam_pose_world

    return (h, Hx) if not get_global_cam_pose else (h, Hx, cam_pose_world)


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
