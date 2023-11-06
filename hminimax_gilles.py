from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance

class PacmanAgent(Agent):
    """Pacman agent based on minimax adversial search."""

    def __init__(self):
        super().__init__()
        self.initial_state = None  # To retrieve initial nb of food dots
        self.depth = 3

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move."""
        if self.initial_state is None:
            self.initial_state = state

        can_win_next_move, action = self.is_next_win(state)
        if can_win_next_move:
            return action

        _, next_move = self.minimax(state)
        return next_move

    def heuristic_function(self, state):
        """
        Evaluate the heuristic value of a state.

        :param state: The game state to evaluate.
        :return: A numeric value representing the heuristic score of the state.
        """
        score = state.getScore()

        # Get positions of Pacman and the ghosts
        pacman_pos = state.getPacmanPosition()
        ghost_positions = [state.getGhostPosition(i) for i in range(1, state.getNumAgents())]

        # Adjust score based on distance to ghosts
        ghost_distances = [manhattanDistance(pacman_pos, ghost_pos) for ghost_pos in ghost_positions]
        for dist in ghost_distances:
            if dist > 0:
                # The further the ghosts, the better, but the effect diminishes with closer distance
                score += 10 / dist

        # Adjust score based on remaining food
        food_list = state.getFood().asList()
        remaining_food = len(food_list)
        score -= 2 * remaining_food

        # Adjust score based on distance to the closest food dot
        if food_list:
            closest_food_dist = min(manhattanDistance(pacman_pos, food) for food in food_list)
            score -= closest_food_dist

        # # Adjust score based on sum of distances to all food dots
        # if food_list:
        #     food_distances = [manhattanDistance(pacman_pos, food) for food in food_list]
        #     sum_food_distances = sum(food_distances)
        #     score -= sum_food_distances


        return score

    def minimax(self, state, depth=0, agentIndex=0, alpha=float('-inf'), beta=float('inf')):
        """Minimax algorithm with alpha-beta pruning and heuristic evaluation for Pacman game.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: the current depth of the search tree.
            agentIndex: the index of the current agent.
            alpha: the best value that the maximizing player can guarantee.
            beta: the best value that the minimizing player can guarantee.

        Return:
            A tuple containing the best score value and corresponding action.
        """
        # Terminal state or max depth
        if state.isWin() or state.isLose():
            return self.heuristic_function(state), Directions.STOP

        # Cut-off depth reached, use heuristic evaluation
        if depth == self.depth:
            return self.heuristic_function(state), Directions.STOP

        # Maximizing for Pacman
        if agentIndex == 0:
            return self.max_value(state, depth, agentIndex, alpha, beta)
        # Minimizing for ghosts
        else:
            return self.min_value(state, depth, agentIndex, alpha, beta)



    def max_value(self, state, depth, agentIndex, alpha, beta):
        """ Returns the maximum value and corresponding action 
        for the given state and agent.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: the current depth of the search tree.
            agentIndex: the index of the current agent.
            alpha: the current alpha value for a-B pruning.
            beta: the current beta value for a-B pruning.

        Return:
            A tuple containing the maximum value and corresponding action.
        """
        value = float('-inf')
        best_action = Directions.STOP
        for s, a in state.generatePacmanSuccessors():
            if agentIndex == state.getNumAgents() - 1:
                new_value, _ = self.minimax(s, depth + 1, 0, alpha, beta)
            else:
                new_value, _ = self.minimax(s, depth, agentIndex + 1, alpha, beta)
            if new_value > value:
                value, best_action = new_value, a
            if value > beta:
                return value, best_action
            alpha = max(alpha, value)
        return value, best_action

    def min_value(self, state, depth, agentIndex, alpha, beta):
        """ Returns the minimum value and corresponding action 
        for the given state and agent.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: the current depth of the search tree.
            agentIndex: the index of the current agent.
            alpha: the current alpha value for a-B pruning.
            beta: the current beta value for a-B pruning.

        Return:
            A tuple containing the minimum value and corresponding action.
        """
        value = float('inf')
        for s, a in state.generateGhostSuccessors(agentIndex):
            if agentIndex == state.getNumAgents() - 1:
                new_value, _ = self.minimax(s, depth + 1, 0, alpha, beta)
            else:
                new_value, _ = self.minimax(s, depth, agentIndex + 1, alpha, beta)
            if new_value < value:
                value = new_value
            if value < alpha:
                return value, Directions.STOP
            beta = min(beta, value)
        return value, Directions.STOP

    def utility_function(self, state):
        """Calculates the utility score of a given game state.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            The utility score of the given game state.
        """
        score = 0
        if state.isWin():
            score += 500
        elif state.isLose():
            score -= 500

        # Number of eaten food dots + time steps
        score += 10 * (self.initial_state.getNumFood() - state.getNumFood())
        score -= state.getScore()

        # Penalize encounters with ghosts
        pacman_pos = state.getPacmanPosition()
        for i in range(1, state.getNumAgents()):
            ghost_dist = manhattanDistance(pacman_pos,state.getGhostPosition(i))
            if ghost_dist < 2:  # If the ghost is too close
                score -= 200

        # Prioritize going to the closest food dot
        food_list = state.getFood().asList()
        if food_list:
            closest_food_dist = min(
                manhattanDistance(pacman_pos, food) for food in food_list
            )
            score -= closest_food_dist

        return score

    def is_next_win(self, state):
        """Check if Pacman can win in the next move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A tuple containing:
            - boolean indicating whether Pacman can win in the next move,
            - action that would lead to a win if it exists, or None otherwise.
        """
        for action in state.getLegalActions(0):  # 0 for Pacman
            successor = state.generatePacmanSuccessor(action)
            if successor.isWin():
                return True, action
        return False, None