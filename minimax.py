from pacman_module.game import Agent, Directions
from pacman_module.pacman import GameState


class PacmanAgent(Agent):   
    """Pacman agent based on minimax adversial search."""

    def __init__(self):
        super().__init__()
        self.initial_state = None
        self.next_move = None
        self.depth = 3
        
    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """
        if self.initial_state is None:
            self.initial_state = state

        self.next_move = self.minimax(state)
        return self.next_move
        
    def minimax(self, state, depth=0, agentIndex=0):
        # Terminal state or max depth
        if state.isWin() or state.isLose() or depth == self.depth:
            return Directions.STOP

        # Check if all agents have had their turn this depth
        # If so, start a new depth
        if agentIndex == state.getNumAgents(): 
            return self.minimax(state, depth + 1, 0)

        # Else, continue with current depth
        else:
            actions = state.getLegalActions(agentIndex)
            if len(actions) == 0:  # No more legal actions for this agent
                return Directions.STOP

            successors = []
            if agentIndex == 0:  # Pacman
                successors = [(s, self.minimax(s, depth, agentIndex + 1), a) for s, a in state.generatePacmanSuccessors()]
            else:  # Ghost
                successors = [(s, self.minimax(s, depth, agentIndex + 1), a) for s, a in state.generateGhostSuccessors(agentIndex)]
            
            if agentIndex == 0:  # Pacman
                return max(successors, key=lambda x: self.utilityFunction(x[0]))[2]
            else:  # Ghost
                return min(successors, key=lambda x: self.utilityFunction(x[0]))[2]

    def utilityFunction(self, state):
        score = 0
        if state.isWin():
            score += 500
        elif state.isLose():
            score -= 500

        score += 10 * (self.initial_state.getNumFood() - state.getNumFood()) # Number of eaten food dots
        score -= 5 * len(state.getCapsules())  # Number of remaining capsules
        score -= state.getScore()  # Time steps

        return score