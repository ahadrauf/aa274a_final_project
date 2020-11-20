import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from utils import plot_line_segments



class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.occupancy = occupancy                      # obstacle set (line segments)
        self.resolution = resolution
        self.path = None        # the final path as a list of states

    def is_free_motion(self, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRT")
        
    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        raise NotImplementedError("find_nearest must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")
        
    def generate_path(self, V, P, n):
        """
        A custom function I wrote for generating the path trajectory given V and P
        V[n,:] = the goal state
        """
        path = []
        current_idx = n
        while current_idx != -1:
            path.append(V[current_idx, :])
            current_idx = P[current_idx]
        return np.flip(np.reshape(path, (len(path), -1)), axis=0)
        

    def solve(self, eps=0.12, max_iters=1000, goal_bias=0.05, shortcut=False):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """

        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        V = np.zeros((max_iters, state_dim))
        V[0,:] = self.x_init    # RRT is rooted at self.x_init
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])

        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root, in general 0 <= P[i] < i for all i < n
        P = -np.ones(max_iters, dtype=int)

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V, P, n: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of states (tree nodes)
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.

        ## Hints:
        #   - use the helper functions find_nearest, steer_towards, and is_free_motion
        #   - remember that V and P always contain max_iters elements, but only the first n
        #     are meaningful! keep this in mind when using the helper functions!

        ########## Code starts here ##########
        success = False
        for _ in range(max_iters):
            if np.random.uniform() < goal_bias:
                x_rand = self.x_goal
            else:
                x_rand = np.random.uniform(self.statespace_lo, self.statespace_hi)
            
            x_near_idx = self.find_nearest(V[range(n),:], x_rand)
            x_near = V[x_near_idx,:]
            x_new = self.steer_towards(x_near, x_rand, eps)
            
            if self.is_free_motion(x_near, x_new):
                V[n,:] = x_new
                P[n] = x_near_idx
                if np.all(x_new == self.x_goal):
                    self.path = self.generate_path(V, P, n)
                    success = True
                    break
                n += 1
                
        ########## Code ends here ##########

        # plt.figure()
        # self.plot_problem()
        # self.plot_tree(V, P, color="blue", linewidth=.5, label="RRT tree", alpha=0.5)
            # if success:
            #     if shortcut:
            #         self.plot_path(color="purple", linewidth=2, label="Original solution path")
            #         self.shortcut_path()
            #         self.plot_path(color="green", linewidth=2, label="Shortcut solution path")
            #     else:
            #         self.plot_path(color="green", linewidth=2, label="Solution path")
            #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
            #     plt.scatter(V[:n,0], V[:n,1])
            # else:
            #     print "Solution not found!"
        print("Solution", success, self.path)

        return success

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        plt.axis('scaled')

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        Input:
            None
        Output:
            None, but should modify self.path
        """
        ########## Code starts here ##########
        success = False
        while not success:
            success = True
            idx = 1
            while idx < len(self.path) - 1:                    
                if self.is_free_motion(self.path[idx-1,:], self.path[idx+1,:]):
                    self.path = np.delete(self.path, idx, 0)
                    success = False
                else:
                    idx += 1
        ########## Code ends here ##########

class AStar(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest(self, V, x):
        ########## Code starts here ##########
        # Hint: This should take one line.
        return min(range(len(V)), key=lambda idx: np.linalg.norm(V[idx,:] - x))
        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        ########## Code starts here ##########
        # Hint: This should take one line.
        if np.linalg.norm(x1 - x2) < eps:
            return x2
        else:
            return x1 + (x2 - x1)*eps/np.linalg.norm(x1 - x2)
        ########## Code ends here ##########

    def is_free(self, x):
        """
        Checks if a give state is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: look at the usage for the DetOccupancyGrid2D.is_free() method
        """
        ########## Code starts here ##########
        # Check to make sure the point is within the bounds of the map
        for idx, val in enumerate(x):
            if val < self.statespace_lo[idx] or val > self.statespace_hi[idx]:
                return False
        return self.occupancy.is_free(x)
        ########## Code ends here ##########

    def is_free_motion(self, x1, x2):
        for pt in np.linspace(x1, x2):
            if not self.is_free(self.snap_to_grid(pt)):
                return False
        return True

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return np.array([self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution)])

    def is_free(self, x):
        """
        Checks if a give state is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: look at the usage for the DetOccupancyGrid2D.is_free() method
        """
        ########## Code starts here ##########
        # Check to make sure the point is within the bounds of the map
        for idx, val in enumerate(x):
            if val < self.statespace_lo[idx] or val > self.statespace_hi[idx]:
                return False
        return self.occupancy.is_free(x)
        ########## Code ends here ##########

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles"""
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        for obs in self.obstacles:
            ax = fig.add_subplot(111, aspect='equal')
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))
