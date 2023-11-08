from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):
    """Pacman agent based on heuristic minimax adversarial search."""

    def __init__(self):
        super().__init__()
        self.initial_state = None   # To retrieve initial nb of food dots
        self.depth = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Args:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        if self.initial_state is None and self.depth is None:
            self.initial_state = state
            self.scale_depth()

        can_win_next_move, action = self.is_next_win(state)
        if can_win_next_move:
            return action

        _, next_move = self.hminimax(state)
        return next_move

    def scale_depth(self):
        """ Scales the depth of the search tree based on the size of the maze.

        If the maze size is <= 100, the depth is set to 1.
        Otherwise, the depth is set to 4.

        Args: None
        Returns: None
        """
        # Determine the size of the maze
        maze_size = self.initial_state.getFood().width \
            * self.initial_state.getFood().height

        # Define threshold to distinguish small-medium from large mazes
        small_threshold = 7 * 7
        medium_threshold = 9 * 9

        # remaining_food = state.getNumFood()
        if maze_size <= small_threshold:
            self.depth = 1
        elif maze_size <= medium_threshold:
         # Adjustments for small maze
            self.depth = 2
        else:
            # Adjustments for large maze
            self.depth = 4

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

    def hminimax(self, s, depth=0, agent=0, a=float('-inf'), b=float('inf')):
        """H-Minimax algorithm with alpha-beta pruning and
        heuristic evaluation for Pacman game.

        Args:
            state: a game state. See API or class `pacman.GameState`.
            depth: the current depth of the search tree.
            agentIndex: the index of the current agent.
            alpha: the best value that the maximizing player can guarantee.
            beta: the best value that the minimizing player can guarantee.

        Returns:
            A tuple containing the best score value and corresponding action.
        """
        # Terminal state or max depth = Cut-off test
        if self.cutoff(depth) or self.terminal(s):
            return self.eval(s), Directions.STOP

        if agent == 0:  # Pacman's turn (Maximizing player)
            return self.max_value(s, depth, agent, a, b)
        else:   # Ghosts' turn (Minimizing player)
            return self.min_value(s, depth, agent, a, b)

    def cutoff(self, depth):
        """Determine if the search should be stopped at this depth.

        Args:
            depth (int): The current depth of the search.

        Returns:
            bool: True if the search should be stopped, False otherwise.
        """
        return depth >= self.depth

    def terminal(self, state):
        """Determines whether the given state is a terminal state,
        i.e. whether the game is over.

        Args:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            True if the game is over (either win or lose), False otherwise.
        """
        return state.isWin() or state.isLose()

    def eval(self, state):
        """Estimates the expected utility of the game from a given state.

        Args:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            An estimate of the expected utility from the given game state.
        """
        # Expected utility estimate
        evaluation = state.getScore()

        # Get positions of Pacman and the ghosts
        pacman_pos = state.getPacmanPosition()
        ghost_positions = [
            state.getGhostPosition(i) for i in range(1, state.getNumAgents())
        ]

        # Adjust evaluation based on distance to ghosts
        ghost_distances = [
            manhattanDistance(
                pacman_pos, ghost_pos
            ) for ghost_pos in ghost_positions
        ]
        for dist in ghost_distances:
            if dist > 0:
                # The further the ghosts, the better,
                # but the effect diminishes with closer distance
                evaluation += 10 / dist

        # Adjust evaluation based on remaining food
        evaluation -= 2 * state.getNumFood()

        # Adjust evaluation based on distance to the closest food dot
        food_list = state.getFood().asList()
        if food_list:
            closest_food_dist = min(
                manhattanDistance(pacman_pos, food) for food in food_list
            )
            evaluation -= 3 * closest_food_dist

        return evaluation

    def max_value(self, state, depth, agentIndex, alpha, beta):
        """Returns the maximum value and corresponding action
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

        # Generate successors and order them by heuristic value
        successors = state.generatePacmanSuccessors()
        ordered_successors = sorted(
            successors, key=lambda x: self.eval(x[0]), reverse=True
        )

        for s, a in ordered_successors:
            if agentIndex == state.getNumAgents() - 1:
                new_value, _ = self.hminimax(s, depth + 1, 0, alpha, beta)
            else:
                new_value, _ = self.hminimax(
                    s, depth, agentIndex + 1, alpha, beta
                )
            if new_value > value:
                value, best_action = new_value, a
            if value > beta:
                return value, best_action
            alpha = max(alpha, value)
        return value, best_action

    def min_value(self, state, depth, agentIndex, alpha, beta):
        """Returns the minimum value and corresponding action
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
        best_action = Directions.STOP

        # Generate successors and order them by heuristic value
        successors = state.generateGhostSuccessors(agentIndex)
        ordered_successors = sorted(
            successors, key=lambda x: self.eval(x[0])
        )

        for s, a in ordered_successors:
            if agentIndex == state.getNumAgents() - 1:
                new_value, _ = self.hminimax(s, depth + 1, 0, alpha, beta)
            else:
                new_value, _ = self.hminimax(
                    s, depth, agentIndex + 1, alpha, beta
                )
            if new_value < value:
                value, best_action = new_value, a
            if value < alpha:
                return value, best_action
            beta = min(beta, value)
        return value, best_action
