import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from utils import plot_line_segments


class AStar(object):
    """Represents a motion planning problem to be solved using RRT"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        print(self.x_init)
        print(self.x_goal)

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.path = None        # the final path as a list of states

    def find_nearest_forward(self, V, x):
        ########## Code starts here ##########
        # Hint: This should take one line.
        return min(range(len(V)), key=lambda idx: np.linalg.norm(V[idx,:] - x))
        ########## Code ends here ##########

    def find_nearest_backward(self, V, x):
        return self.find_nearest_forward(V, x)

    def steer_towards_forward(self, x1, x2, eps):
        ########## Code starts here ##########
        # Hint: This should take one line.
        if np.linalg.norm(x1 - x2) < eps:
            return self.snap_to_grid(x2)
        else:
            return self.snap_to_grid(x1 + (x2 - x1)*eps/np.linalg.norm(x1 - x2))
        ########## Code ends here ##########

    def steer_towards_backward(self, x1, x2, eps):
        return self.steer_towards_forward(x2, x1, eps)

    def is_free_motion(self, x1, x2):
        for pt in np.linspace(x1, x2):
            if not self.is_free(self.snap_to_grid(pt)):
                return False
        return True

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

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line.
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return np.array([self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution)])

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by simply adding/subtracting self.resolution
               from x, numerical error could creep in over the course of many
               additions and cause grid point equality checks to fail. To remedy
               this, you should make sure that every neighbor is snapped to the
               grid as it is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        # Start at angle = 0 and go counter clockwise (right hand rule)
        dxs = np.array([1, 1, 0, -1, -1, -1, 0, 1]) * self.resolution
        dys = np.array([0, 1, 1, 1, 0, -1, -1, -1]) * self.resolution
        for dx, dy in zip(dxs, dys):
            neighbor = (self.snap_to_grid(x + np.array([dx, dy])))
            if self.is_free(neighbor):
                neighbors.append(neighbor)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def generate_path(self, V_fw, P_fw, n_fw, V_bw, P_bw, n_bw):
        """
        A custom function I wrote for generating the path trajectory given V and P
        V[n,:] = the goal state
        """
        path_forward = []
        current_idx = n_fw - 1
        while current_idx != -1:
            path_forward.append(V_fw[current_idx, :])
            current_idx = P_fw[current_idx]
        path_forward = np.flip(np.reshape(path_forward, (len(path_forward), -1)), axis=0)
            
        current_idx = 0 # n_bw
        path_backward = []
        while current_idx != -1:
            path_backward.append(V_bw[current_idx, :])
            current_idx = P_bw[current_idx]
        path_backward = np.flip(np.reshape(path_backward, (len(path_backward), -1)), axis=0)
        return np.vstack([path_forward, path_backward])

    def solve(self, eps=0.08, max_iters = 1000):
        """
        Uses RRT-Connect to perform bidirectional RRT, with a forward tree
        rooted at self.x_init and a backward tree rooted at self.x_goal, with
        the aim of producing a dynamically-feasible and obstacle-free trajectory
        from self.x_init to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
                
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """
        
        state_dim = len(self.x_init)

        V_fw = np.ones((max_iters, state_dim))*-np.Inf     # Forward tree
        V_bw = np.ones((max_iters, state_dim))*-np.Inf     # Backward tree

        n_fw = 1    # the current size of the forward tree
        n_bw = 1    # the current size of the backward tree

        P_fw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the forward tree
        P_bw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the backward tree

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V_fw, V_bw, P_fw, P_bw, n_fw, n_bw: the represention of the
        #           planning trees
        #    - success: whether or not you've found a solution within max_iters
        #           RRT-Connect iterations
        #    - self.path: if success is True, then must contain list of states
        #           (tree nodes) [x_init, ..., x_goal] such that the global
        #           trajectory made by linking steering trajectories connecting
        #           the states in order is obstacle-free.
        # Hint: Use your implementation of RRT as a reference

        ########## Code starts here ##########
        V_fw[0,:] = self.x_init
        V_bw[0,:] = self.x_goal
        print(n_fw, n_bw)
        
        success = False
        for cnt in range(max_iters // 2):
            if (cnt % 25) == 0:
                print("Currently at iteration " + str(cnt) + " of RRT planning")
                print(V_fw)
                print(V_bw)
            if success:
                break
            idx = np.random.randint(n_fw)
            
            x_rand = self.snap_to_grid(np.clip(np.array([np.random.normal(V_fw[idx, 0]), 
                np.random.normal(V_fw[idx, 1])]), self.statespace_lo, self.statespace_hi))
            # x_rand = self.snap_to_grid(np.random.uniform(self.statespace_lo, self.statespace_hi))
            x_near_idx = self.find_nearest_forward(V_fw[range(n_fw),:], x_rand)
            x_near = V_fw[x_near_idx,:]
            x_new = self.steer_towards_forward(x_near, x_rand, eps)
            
            if self.is_free_motion(x_near, x_new):
                V_fw[n_fw,:] = x_new
                P_fw[n_fw] = x_near_idx
                n_fw += 1
                
                x_connect_idx = self.find_nearest_backward(V_bw[range(n_bw),:], x_new)
                x_connect = V_bw[x_connect_idx,:]
                while True:
                    x_newconnect = self.steer_towards_backward(x_new, x_connect, eps)
#                     x_newconnect = self.steer_towards_backward(x_connect, x_new, eps)
                    if self.is_free_motion(x_newconnect, x_connect):
                        V_bw[n_bw,:] = x_newconnect
                        P_bw[x_connect_idx] = n_bw
                        n_bw += 1
                        
                        if np.all(x_newconnect == x_new):
                            self.path = self.generate_path(V_fw[range(n_fw),:], 
                                                           P_fw[range(n_fw)], 
                                                           n_fw, 
                                                           V_bw[range(n_bw),:], 
                                                           P_bw[range(n_bw)], 
                                                           n_bw)
                            success = True
                            break
                        x_connect = x_newconnect
                    else:
                        break
            if success:
                break
                
            # Backward pass
            idx = np.random.randint(n_bw)
            
            x_rand = self.snap_to_grid(np.clip(np.array([np.random.normal(V_bw[idx, 0]), 
                np.random.normal(V_bw[idx, 1])]), self.statespace_lo, self.statespace_hi))
            # x_rand = self.snap_to_grid(np.random.uniform(self.statespace_lo, self.statespace_hi))
            x_near_idx = self.find_nearest_backward(V_bw[range(n_bw),:], x_rand)
            x_near = V_bw[x_near_idx,:]
            x_new = self.steer_towards_backward(x_rand, x_near, eps)
#             x_new = self.steer_towards_backward(x_near, x_rand, eps)
            
            if self.is_free_motion(x_new, x_near):
                V_bw[n_bw,:] = x_new
#                 P_bw[n_bw] = x_near_idx
                P_bw[x_near_idx] = n_bw
                n_bw += 1
                
                x_connect_idx = self.find_nearest_forward(V_fw[range(n_fw),:], x_new)
                x_connect = V_fw[x_connect_idx,:]
                while True:
                    x_newconnect = self.steer_towards_forward(x_connect, x_new, eps)
                    if self.is_free_motion(x_connect, x_newconnect):
                        V_fw[n_fw,:] = x_newconnect
                        P_fw[n_fw] = x_connect_idx
                        n_fw += 1                        
                        if np.all(x_newconnect == x_new):
                            self.path = self.generate_path(V_fw[range(n_fw),:], 
                                                           P_fw[range(n_fw)], 
                                                           n_fw, 
                                                           V_bw[range(n_bw),:], 
                                                           P_bw[range(n_bw)], 
                                                           n_bw)
                            success = True
                            break
                        x_connect = x_newconnect
                    else:
                        break

        ########## Code ends here ##########

        # plt.figure()
        # self.plot_problem()
        # self.plot_tree(V_fw, P_fw, color="blue", linewidth=.5, label="RRTConnect forward tree")
        # self.plot_tree_backward(V_bw, P_bw, color="purple", linewidth=.5, label="RRTConnect backward tree")
        
        # if success:
        #     self.plot_path(color="green", linewidth=2, label="solution path")
        #     plt.scatter(V_fw[:n_fw,0], V_fw[:n_fw,1], color="blue")
        #     plt.scatter(V_bw[:n_bw,0], V_bw[:n_bw,1], color="purple")
        # plt.scatter(V_fw[:n_fw,0], V_fw[:n_fw,1], color="blue")
        # plt.scatter(V_bw[:n_bw,0], V_bw[:n_bw,1], color="purple")

        # plt.show()


        print(self.path, success)
        self.shortcut_path()
        print('Shortened path', self.path)
        return success

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

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
            if self.path is not None:
                while idx < len(self.path) - 1:                    
                    if self.is_free_motion(self.path[idx-1,:], self.path[idx+1,:]):
                        self.path = np.delete(self.path, idx, 0)
                        success = False
                    else:
                        idx += 1
        ########## Code ends here ##########

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
