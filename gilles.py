from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance

class PacmanAgent(Agent):
    def __init__(self):
        self.depth = 4

    def look_ahead_for_win(self, state):
    # Check if Pacman can win in the next move
        for action in state.getLegalActions(0):  # 0 for Pacman
            successor = state.generatePacmanSuccessor(action)
            if successor.isWin():
                return True, action
        return False, None

    def get_action(self, state):
        """
        Given a Pacman game state, returns a legal move by applying the Minimax algorithm.
        """
        can_win_next_move, action = self.look_ahead_for_win(state)
        if can_win_next_move:
            return action

        action, _ = self.minimax(state, self.depth, 0)
        return action


    def minimax(self, state, depth, agentIndex):
        """
        Performs the Minimax algorithm, considering both Pacman (agentIndex=0) and the ghosts.
        """
        if state.isWin() or state.isLose() or depth == 0:
            return Directions.STOP, self.evaluation_function(state)

        if agentIndex == 0:  # Pacman's turn (Maximizing player)
            return self.max_value(state, depth, agentIndex)
        else:  # Ghosts' turn (Minimizing player)
            return self.min_value(state, depth, agentIndex)


    def max_value(self, state, depth, agentIndex):
        """
        Computes the max value for the Minimax algorithm (Pacman's perspective).
        """
        best_value = float('-inf')
        best_action = Directions.STOP

        for action in state.getLegalActions(agentIndex):
            if action == Directions.STOP:
                continue  # Skip the STOP action
            successor = state.generateSuccessor(agentIndex, action)
            _, value = self.minimax(successor, depth, (agentIndex + 1) % state.getNumAgents())
            if value > best_value:
                best_value = value
                best_action = action

        return best_action, best_value

    def min_value(self, state, depth, agentIndex):
        """
        Computes the min value for the Minimax algorithm (Ghosts' perspective).
        """
        best_value = float('inf')
        best_action = Directions.STOP

        for action in state.getLegalActions(agentIndex):
            if action == Directions.STOP:
                continue  # Skip the STOP action
            successor = state.generateSuccessor(agentIndex, action)
            next_agent = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth - 1 if next_agent == 0 else depth
            _, value = self.minimax(successor, next_depth, next_agent)
            if value < best_value:
                best_value = value
                best_action = action

        return best_action, best_value

    def evaluation_function(self, state):
        """
        The evaluation function for the current state.
        Prioritizes eating food and staying alive over ghost avoidance.
        """
        if state.isWin():
            return float('inf')  # Maximize score for winning state
        if state.isLose():
            return float('-inf')  # Minimize score for losing state

        pacman_pos = state.getPacmanPosition()
        food_list = state.getFood().asList()
        ghost_positions = [state.getGhostPosition(i) for i in range(1, state.getNumAgents())]

        # Calculate the distance to the nearest food dot
        min_food_distance = min([manhattanDistance(pacman_pos, food) for food in food_list])
        # Find the closest food position
        closest_food_pos = min(food_list, key=lambda x: manhattanDistance(pacman_pos, x))

        # Calculate the distance to the nearest ghost
        min_ghost_distance = min([manhattanDistance(pacman_pos, ghost_pos) for ghost_pos in ghost_positions])
        # Find the closest ghost position
        closest_ghost_pos = min(ghost_positions, key=lambda x: manhattanDistance(pacman_pos, x))

        # Check if Pacman is between the ghost and the food
        if manhattanDistance(pacman_pos, closest_ghost_pos) > manhattanDistance(closest_food_pos, closest_ghost_pos):
            # Pacman is closer to the food than the ghost is, prioritize getting the food
            min_food_distance = 0  # This will give the highest priority to getting the food

        # Adjust the score based on the proximity of food and ghosts
        score = state.getScore()
        score += 10 * (1 / (min_food_distance + 1))  # Prioritize closer food more

        # If the ghost is too close, heavily penalize the state to avoid getting eaten
        if min_ghost_distance <= 1:
            score -= 500  # Large penalty for being close to a ghost

        return score


