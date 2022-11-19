# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        d = []
        foodList = currentGameState.getFood().asList()
        pacman_position = list(newPos)

        if action == 'Stop':
            return -float("inf")

        for ghostState in newGhostStates:
            if ghostState.getPosition() == tuple(pacman_position) and ghostState.scaredTimer == 0:
                return -float("inf")

        for food in foodList:
            x = abs(food[0] - pacman_position[0]) * (-1)
            y = abs(food[1] - pacman_position[1]) * (-1)
            d.append(x + y)

        ans = max(d)
        return ans


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        result = self.getValue(gameState, 0, 0)
        return result[1]

    def getValue(self, gameState, index, depth):
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), ""

        # Pacman has index = 0
        if index == 0:
            return self.maxValue(gameState, index, depth)

        # Ghost has index > 0
        else:
            return self.minValue(gameState, index, depth)

    def maxValue(self, gameState, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = gameState.getLegalActions(index)
        max_val = float("-inf")
        max_act = ""

        for a in legalMoves:
            successor = gameState.generateSuccessor(index, a)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.getValue(
                successor, successor_index, successor_depth)[0]

            if current_value > max_val:
                max_val = current_value
                max_act = a

        return max_val, max_act

    def minValue(self, gameState, index, depth):
        """
        Returns the min utility value-action for min-agent
        """
        legalMoves = gameState.getLegalActions(index)
        min_val = float("inf")
        min_act = ""

        for a in legalMoves:
            successor = gameState.generateSuccessor(index, a)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_value = self.getValue(
                successor, successor_index, successor_depth)[0]

            if current_value < min_val:
                min_val = current_value
                min_act = a

        return min_val, min_act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        result = self.getBestActionAndScore(
            gameState, 0, 0, float("-inf"), float("inf"))
        return result[0]

    def getBestActionAndScore(self, gameState, index, depth, alpha, beta):
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "", gameState.getScore()

        # Pacman has index = 0
        if index == 0:
            return self.maxValue(gameState, index, depth, alpha, beta)

        # Ghost has index > 0
        else:
            return self.minValue(gameState, index, depth, alpha, beta)

    def maxValue(self, gameState, index, depth, alpha, beta):
        legalMoves = gameState.getLegalActions(index)
        max_val = float("-inf")
        max_act = ""

        for a in legalMoves:
            successor = gameState.generateSuccessor(index, a)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            if current_value > max_val:
                max_val = current_value
                max_act = a

            alpha = max(alpha, max_val)

            if max_val > beta:
                return max_act, max_val

        return max_act, max_val

    def minValue(self, game_state, index, depth, alpha, beta):
        legalMoves = game_state.getLegalActions(index)
        min_val = float("inf")
        min_act = ""

        for a in legalMoves:
            successor = game_state.generateSuccessor(index, a)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            if current_value < min_val:
                min_val = current_value
                min_act = a

            beta = min(beta, min_val)
            if min_val < alpha:
                return min_act, min_val

        return min_act, min_val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        action, score = self.getValue(gameState, 0, 0)
        return action

    def getValue(self, gameState, index, depth):
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "", self.evaluationFunction(gameState)

        # Pacman has index = 0
        if index == 0:
            return self.maxValue(gameState, index, depth)

        # Ghost has index > 0
        else:
            return self.expectedValue(gameState, index, depth)

    def maxValue(self, gameState, index, depth):
        legalMoves = gameState.getLegalActions(index)
        max_val = float("-inf")
        max_act = ""

        for a in legalMoves:
            successor = gameState.generateSuccessor(index, a)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_action, current_value = self.getValue(
                successor, successor_index, successor_depth)

            if current_value > max_val:
                max_val = current_value
                max_act = a

        return max_act, max_val

    def expectedValue(self, game_state, index, depth):
        legalMoves = game_state.getLegalActions(index)
        expected_val = 0
        expected_act = ""

        successor_probability = 1.0 / len(legalMoves)

        for a in legalMoves:
            successor = game_state.generateSuccessor(index, a)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_action, current_value = self.getValue(
                successor, successor_index, successor_depth)

            expected_val += successor_probability * current_value

        return expected_act, expected_val


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Function will find the distance of the pacman to all food,
    then find the distance between the pacman and all ghosts, then if the 
    ghost is too near the pacman, we will prioritize escaping from eating
    the food. We will combine the features linearly and return the sum
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    ghost_pos = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    closest_food = 1

    game_score = currentGameState.getScore()

    food_distances = [manhattanDistance(
        pacman_pos, food_position) for food_position in food_list]

    if food_count > 0:
        closest_food = min(food_distances)

    for p in ghost_pos:
        ghost_distance = manhattanDistance(pacman_pos, p)

        if ghost_distance < 2:
            closest_food = 99999

    features = [1.0 / closest_food, game_score, food_count, capsule_count]

    weights = [10, 200, -100, -10]
    result = sum([feature * weight for feature, weight in zip(features, weights)])
    return result


# Abbreviation
better = betterEvaluationFunction
