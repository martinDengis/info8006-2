from pacman_module.game import Agent, Directions
from pacman_module.pacman import GameState
from pacman_module.util import manhattanDistance


class PacmanAgent(Agent):   
    """Pacman agent based on minimax adversial search."""

    def __init__(self):
        super().__init__()
        self.initial_state = None
        self.next_move = None
        self.depth = 4
        
    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """
        if self.initial_state is None:
            self.initial_state = state

        can_win_next_move, action = self.look_ahead_for_win(state)
        if can_win_next_move:
            return action

        _, self.next_move = self.minimax(state)
        return self.next_move
        
    def minimax(self, state, depth=0, agentIndex=0, alpha=float('-inf'), beta=float('inf')):
        # Terminal state or max depth
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.utilityFunction(state), Directions.STOP

        # Check if all agents have had their turn this depth
        # If so, start a new depth
        if agentIndex == state.getNumAgents(): 
            return self.minimax(state, depth + 1, 0, alpha, beta)

        # Else, continue with current depth
        else:
            actions = state.getLegalActions(agentIndex)
            if len(actions) == 0:  # No more legal actions for this agent
                return self.utilityFunction(state), Directions.STOP

            if agentIndex == 0:  # Pacman
                value = float('-inf')
                best_action = Directions.STOP
                for s, a in state.generatePacmanSuccessors():
                    new_value, _ = self.minimax(s, depth, agentIndex + 1, alpha, beta)
                    if new_value > value:
                        value, best_action = new_value, a
                    if value > beta:
                        return value, best_action
                    alpha = max(alpha, value)
                return value, best_action
            else:  # Ghost
                value = float('inf')
                for s, a in state.generateGhostSuccessors(agentIndex):
                    new_value, _ = self.minimax(s, depth, agentIndex + 1, alpha, beta)
                    if new_value < value:
                        value = new_value
                    if value < alpha:
                        return value, Directions.STOP
                    beta = min(beta, value)
                return value, Directions.STOP

    def utilityFunction(self, state):
        score = 0
        if state.isWin():
            score += 500
        elif state.isLose():
            score -= 500

        score += 10 * (self.initial_state.getNumFood() - state.getNumFood())  # Number of eaten food dots
        score -= state.getScore()  # Time steps

        # Penalize encounters with ghosts
        for i in range(1, state.getNumAgents()):
            ghost_distance = manhattanDistance(state.getPacmanPosition(), state.getGhostPosition(i))
            if ghost_distance < 2:  # If the ghost is too close
                score -= 200  # Subtract a large value from the score

        # Prioritize going to the closest food dot
        food_list = state.getFood().asList()
        if food_list:
            pacman_pos = state.getPacmanPosition()
            closest_food_distance = min(manhattanDistance(pacman_pos, food) for food in food_list)
            score -= closest_food_distance

        return score
    
    def look_ahead_for_win(self, state):
    # Check if Pacman can win in the next move
        for action in state.getLegalActions(0):  # 0 for Pacman
            successor = state.generatePacmanSuccessor(action)
            if successor.isWin():
                return True, action
        return False, None