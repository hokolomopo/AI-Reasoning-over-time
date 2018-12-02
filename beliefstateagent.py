# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
from pacman_module import util


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        """
            Variables to use in 'updateAndFetBeliefStates' method.
            Initialization occurs in 'get_action' method.
        """
        # Current list of belief states over ghost positions
        self.beliefGhostStates = None
        # Grid of walls (assigned with 'state.getWalls()' method) 
        self.walls = None

        self.step = 0

    def updateAndGetBeliefStates(self, evidences):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of (noised) ghost positions at state x_{t}
          where 't' is the current time step

        Return:
        -------
        - A list of Z belief states at state x_{t} about ghost positions
          as N*M numpy matrices of probabilities
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """

        beliefStates = self.beliefGhostStates
        # XXX: Your code here
        height = self.walls.height
        width = self.walls.width
        nbrGhosts = len(evidences)

        w = self.args.w
        p = self.args.p

        oldBeliefState = beliefStates
        beliefStates = np.zeros((nbrGhosts, width, height))

        for k in range(nbrGhosts):
            radar = evidences[k]
            ghostBeliefeState = beliefStates[k]
            
            totalSum = 0

            for i in range(width):
                for j in range(height):
                    proba = 0
                    state = (i, j)
                    lastP = oldBeliefState[k][i][j]
 
                    pEgivenX = self._getProbaEgivenX(radar, state, w)
                    if(pEgivenX == 0):
                        continue

                    pSum = 0
                    for x in range(width):
                        for y in range(height):
                            pSum += self._getProbaXgivenX(state, (x, y), w) + lastP
                                        
                    proba = pEgivenX * pSum
                    ghostBeliefeState[i][j] = proba
                    totalSum += proba
            

            # Don't use np.multiply to not create a new array (and thus improve persormances)
            for i in range(width):
                for j in range(height):
                    ghostBeliefeState[i][j] /= totalSum

            beliefStates[k] = ghostBeliefeState

        # XXX: End of your code
        self.beliefGhostStates = beliefStates
        return beliefStates

    def _getProbaEgivenX(self, e, x, w):
        
        # if (x[0] + w >= e[0]) and  (x[0] - w <= e[0]):
        #     if (x[1] + w >= e[1]) and (x[1] - w <= e[1]):

        if(self._inSquare(e, x, w)):
            return self._getUniformProba(w)
        return 0

    def _getProbaXgivenX(self, x, x_1, p):
        distance = self._compute_manhattan(x_1, x)

        # Impossible to go from x_1 to x in 1 move
        if(distance != 1):
            return 0

        # Go East to go from x_1 to x
        if(x[0] > x_1[0]):
            return p + (1 - p) * 0.25

        return (1 - p) * 0.25

    def _getUniformProba(self, w):
        return 1 / ((w + 2)**2)

    def _computeNoisyPositions(self, state):
        """
            Compute a noisy position from true ghosts positions.
            XXX: DO NOT MODIFY THAT FUNCTION !!!
            Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        w = self.args.w

        div = float(w * w)
        new_positions = []
        for p in positions:
            (x, y) = p
            dist = util.Counter()
            for i in range(x - w, x + w):
                for j in range(y - w, y + w):
                    dist[(i, j)] = 1.0 / div
            new_positions.append(util.chooseFromDistribution(dist))
        return new_positions

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """

        # XXX : You shouldn't care on what is going on below.
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()
        return self.updateAndGetBeliefStates(
            self._computeNoisyPositions(state))

    def _compute_manhattan(self, position1, position2):
        """
        Compute the Manhattan distance beteween 2 positions.

        Arguments:
        ----------
        - `position1`, `position2`: two tuples representing
          positions`.

        Return:
        -------
        - The Manhattan distance between the 2 positions
        """

        return abs(position1[0] - position2[0]) \
            + abs(position1[1] - position2[1])

    def _inSquare(self, x, center, size):
        if (center[0] + size >= x[0]) and  (center[0] - size <= x[0]):
            if (center[1] + size >= x[1]) and (center[1] - size <= x[1]):
                return True

        return False

