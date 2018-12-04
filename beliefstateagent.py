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

        # Get w and p from the arguments given to the program
        w = self.args.w
        p = self.args.p

        # Get height and width of the map
        height = self.walls.height
        width = self.walls.width

        # Get number of ghosts
        nbrGhosts = len(evidences)

        # Remember probability matrix from previous time t
        oldBeliefState = beliefStates

        # Initialise beliefStates with zeros
        beliefStates = np.zeros((nbrGhosts, width, height))

        # Compute a belief matrix for every ghost
        for k in range(nbrGhosts):

            # Get position of the ghost according to the radar
            radar = evidences[k]

            # Keep the sum of all the probability in the matrix to normalize
            # the matrix at the end
            totalPSum = 0

            # Loop for each element of the matrix. Only takes element from
            # index w to size-w to avoid the padding at the border of the map
            for i in range(w, width - w):
                for j in range(w, height - w):

                    # state = X_{t+1} from theory
                    state = (i, j)

                    # Get P(e_{t+1}|x_{t+1})
                    pEgivenX = self._getProbaEgivenX(radar, state, w)
                    if(pEgivenX == 0):
                        continue

                    # Do the sum of probabilities for each x_{t}
                    pSum = 0
                    for x in range(w, width - w):
                        for y in range(w, height - w):
                            # self._getProbaXgivenX() = P(X_{t+1}|x_{t})
                            # oldBeliefState = P(x_{t}|e_{1:t})
                            pSum += self._getProbaXgivenX(state, (x, y), p) *\
                                oldBeliefState[k][x][y]

                    # proba = P(X_{t+1}|e_{1:t+1}), not normalized
                    proba = pEgivenX * pSum
                    beliefStates[k][i][j] = proba
                    totalPSum += proba

            # Normalize the probability matrix to have sum of probas = 1
            # Don't use np.multiply to not create a new array,
            # and thus improve performances
            for i in range(width):
                for j in range(height):
                    beliefStates[k][i][j] /= totalPSum

        # XXX: End of your code
        self.beliefGhostStates = beliefStates
        return beliefStates

    def _getProbaEgivenX(self, e, x, w):
        """
        Given an estimation e, a position of the ghost x and the parameter w,
        returns P(e_{t+1}|x_{t+1})

        Arguments:
        ----------
        - `e`: the estimation of the position of the ghost given by the radar
          for the time t
        - `x`: the position of the ghost for the time t
        - `w`: the parameter of the uniform distribution of the radar

        Return:
        -------
        - The probability of e given x P(e_{t+1}|x_{t+1})
        """

        # If e is in a square centered around x of size w return the
        # probability given by the uniform distribution
        if(self._inSquare(e, x, w)):
            return 1 / ((w + 2)**2)

        # Else returns 0, it's imposible to have e for this x
        return 0

    def _getProbaXgivenX(self, x, x_1, p):
        """
        Given the position of the ghost at time t and t+1, and the parameter p,
        returns P(X_{t+1}|x_{t}), the probability of the ghost being in x in
        t+1 knowing that he was in x_1 in t

        Arguments:
        ----------
        - `x`: the position of the ghost for the time t+1
        - `x_1`: the position of the ghost for the time t
        - `p`: the parameter characterizing the movements of the ghosts

        Return:
        -------
        - The probability P(X_{t+1}|x_{t})
        """

        # Get the Manhattan distance between the 2 points
        distance = self._compute_manhattan(x_1, x)

        # Impossible to go from x_1 to x in 1 move, they are too far appart
        if(distance != 1):
            return 0

        # Go East to go from x_1 to x
        if(x[0] > x_1[0]):
            return p + (1 - p) * 0.25

        # Don't go East to go from x_1 to x
        return (1 - p) * 0.25

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
        """
        Given a point x, returns true if x is in a cquare of center "center"
        and with edges of size "size" + 2

        Arguments:
        ----------
        - `x`: a tuple (x, y) representing a point
        - `center`: a tuple (x, y) representing the center of the square
        - `size`: the distance between the center and the edges of the square

        Return:
        -------
        - True if x is in the square, false otherwise
        """

        if (center[0] + size >= x[0]) and (center[0] - size <= x[0]):
            if (center[1] + size >= x[1]) and (center[1] - size <= x[1]):
                return True

        return False
