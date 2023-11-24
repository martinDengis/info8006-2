from pacman_module.game import Agent
import math

MAX = -math.inf
MIN = math.inf

class Node:
    def __init__(self, state):
        self.state = state
        self.next_nodes = []
        self.actions = []
        self.optimal_action = None
        self.minimaxValue = 0
        self.optimal_next_node = None
        self.information = (state.getPacmanPosition(), state.getGhostPosition(1), state.getNumFood())

    def get_state(self):
        return self.state

    def copy_node(self, cost):
        new_node = Node(self.state)
        new_node.next_nodes = self.next_nodes
        new_node.set_optimal_next_node(self.optimal_next_node)
        new_node.actions = self.actions
        new_node.set_information(self.information)
        new_node.set_optimal_action(self.optimal_action)
        new_node.set_minimax_value(self.minimaxValue - cost)
        return new_node

    def set_information(self, information):
        self.information = information

    def get_information(self):
        return self.information

    def set_optimal_action(self, optimal_action):
        self.optimal_action = optimal_action

    def get_optimal_action(self):
        return self.optimal_action

    def set_next_nodes(self, next_node):
        self.next_nodes.append(next_node)

    def get_next_nodes(self):
        return self.next_nodes

    def set_actions(self, action):
        self.actions.append(action)

    def get_actions(self):
        return self.actions

    def set_minimax_value(self, minimax_value):
        self.minimaxValue = minimax_value

    def get_minimax_value(self):
        return self.minimaxValue

    def set_optimal_next_node(self, optimal_next_node):
        self.optimal_next_node = optimal_next_node

    def get_optimal_next_node(self):
        return self.optimal_next_node


def utility(s, p, depth, fruits_eaten):  # We define it for player p = Pacman
    score = 10 * fruits_eaten - ((depth+1)/2 if depth % 2 != 0 else depth/2)
    if terminal(s) == 1:
        score = score + (500 if s.isWin() else -500)
    return score if p == 1 else -score


def terminal(s):
    return 1 if s.isWin() or s.isLose() else 0


def actions(s, player_turn):
    return s.generatePacmanSuccessors() if player_turn == 1\
        else s.generateGhostSuccessors(1)


def player(depth):   # Return 1 if pacman turn else 0 for ghost turn
    return 1 if depth % 2 == 0 else 0


def get_opt_move(current_node, player_turn):
    pivot = MAX if player_turn == 1 else MIN
    node_successors = current_node.get_next_nodes()
    actions_successors = current_node.get_actions()
    index = 0
    for node in node_successors:
        value_to_compare = node.get_minimax_value()
        action = actions_successors[index]
        if player_turn == 1 and value_to_compare > pivot:
            current_node.set_optimal_action(action)
            current_node.set_minimax_value(value_to_compare)
            current_node.set_optimal_next_node(node)
            pivot = node.get_minimax_value()
        elif player_turn == 0 and value_to_compare < pivot:
            current_node.set_optimal_action(action)
            current_node.set_minimax_value(value_to_compare)
            current_node.set_optimal_next_node(node)
            pivot = value_to_compare
        else:
            pass
        index += 1
    if current_node.get_minimax_value() == 0:
        current_node.set_minimax_value(MIN if player_turn == 1 else MAX)


def is_visited(visited, state):
    for node in visited:
        if node.get_information() == (state.getPacmanPosition(), state.getGhostPosition(1),
                                      state.getNumFood()):
            return node, True
    return None, False


def generate_tree(current_state, depth, initial_fruit_count, visited):

    player_turn = player(depth)

    original_node, already_visited = is_visited(visited, current_state)

    if already_visited is False:
        current_node = Node(current_state)
        visited.append(current_node)
    else:
        new_node = original_node.copy_node((depth+1)/2 if depth % 2 != 0 else depth/2)
        return new_node

    fruit_point = initial_fruit_count - current_state.getNumFood()

    if terminal(current_state) == 1:
        current_node.set_minimax_value(utility(current_state, 1, depth, fruit_point))
        return current_node

    begin_successors = actions(current_state, player_turn)

    for successor in begin_successors:
        next_state = successor[0]
        action = successor[1]
        current_node.set_next_nodes(generate_tree(next_state,
                                                  depth+1, initial_fruit_count, visited))
        current_node.set_actions(action)
    get_opt_move(current_node, player_turn)
    return current_node


def minimax(root):
    return root.get_optimal_next_node(), root.get_optimal_action()


def actualize_root(root, state):
    if root.get_information() == (state.getPacmanPosition(), state.getGhostPosition(1), state.getNumFood()):
        return root

    for next_node in root.get_next_nodes():
        if next_node.get_information() == (state.getPacmanPosition(), state.getGhostPosition(1), state.getNumFood()):
            return next_node


class PacmanAgent(Agent):
    def __init__(self):
        self.root = None
        self.depth = 0
        self.fruit = 0
        self.visited = []
        self.initial_state = None

    def get_action(self, state):
        self.fruit = state.getNumFood()
        self.initial_state = state
        if not self.root:
            self.root = generate_tree(self.initial_state, self.depth, self.fruit, self.visited)
        self.root = actualize_root(self.root, state)
        self.root, move = minimax(self.root)
        return move



