import random
import numpy as np

# defining board
board = np.array([['-','-','-'],['-','-','-'],['-','-','-']])
players = ['X', 'O']
num_players = len(players)
Q = {}

# important parameters
learning_rate = 0.01
discount_factor = 0.9
exploration_rate = 0.5

# training episodes
training_episodes = 10000

# printing the board
def print_board(board):
    for row in board:
        print(' | '.join(row))
        print('-' * 10)
# print_board(board)

# board -> string
def board_to_string(board):
    return ''.join(board.flatten())

# print(board_to_string(board))
empty_cells = np.argwhere(board == '-')
# print(empty_cells)
actions = tuple(random.choice(empty_cells))
# print(actions)
# print(set(board[1]))
# function to check if the game is over
def is_game_over(board):
    # checking for rows
    for row in board:   
        if len(set(row)) == 1 and row[0] != '-':
            return True, row[0]
    # checking for columns
    for col in board.T:
        if len(set(col)) == 1 and col[0] != '-':
            return True, col[0]
    # checking for diagonals
    if (len(set(board.diagonal())) == 1 and board[0][0] != '-'):
        return True, board[0][0]
    if (len(set(np.fliplr(board).diagonal())) == 1 and board[0][2] != '-'):
        return True, board[0][2]
    ## checking for draw
    if '-' not in board:
        return True, 'draw'
    return False, None

# print(is_game_over(board))
def choose_action(board, exploration_rate):
    state = board_to_string(board)

    # exploration
    if random.uniform(0,1) < exploration_rate or state not in Q:
        empty_cells = np.argwhere(board == '-')
        action = tuple(random.choice(empty_cells))
    else:
        q_values = Q[state]
        empty_cells = np.argwhere(board == '-')
        empty_q_values = [q_values[cell[0], cell[1]] for cell in empty_cells]
        max_q_value = max(empty_q_values)
        max_q_indices = [i for i, value in enumerate(empty_q_values) if value == max_q_value]
        max_q_index = random.choice(max_q_indices)
        action = tuple(empty_cells[max_q_index])
    return action

def board_next_state(cell):
    next_state = board.copy()
    next_state[cell[0],cell[1]] = players[0]
    return next_state


def update_q_table(state, action, next_state, reward):
    q_values = Q.get(state,np.zeros((3,3)))
    
    next_q_values = Q.get(board_to_string(next_state) , np.zeros((3,3)))
    max_next_q_value = np.max(next_q_values)

    q_values[action[0], action[1]] += learning_rate * (reward + discount_factor * max_next_q_value - q_values[action[0], action[1]])

    Q[state] = q_values
#main Q learning
num_draws = 0
agent_wins = 0
for episode in range(training_episodes):
    board = np.array([['-','-','-'],['-','-','-'],['-','-','-']])
    current_player = random.choice(players)
    game_over = False
    while not game_over:
        action = choose_action(board, exploration_rate)

        row, col = action
        board[row,col] = current_player

        game_over,winner = is_game_over(board)

        if game_over:
            if winner ==  current_player:
                reward = 1
                agent_wins += 1
            elif winner == 'draw':
                reward = 0.5
                num_draws += 1
            else:
                reward = 0
        else:
            current_player = players[(players.index(current_player) + 1) % num_players]
        if not game_over:
            next_state = board_next_state(action)
            update_q_table(board_to_string(board), action, next_state, reward = 0)
    exploration_rate *= 0.99
    if(episode+1) % 100 == 0:
        print(f"episode {episode+1}/{training_episodes} trained")
        print(f"Current exploration rate: {exploration_rate:.4f}")
        print(f"Q-table-size: ",len(Q))
        print("-"*40)

board = np.array([['-','-','-'],['-','-','-'],['-','-','-']])
current_player = random.choice(players)
game_over = False
def is_valid_move(state,row,col):
    return state[row][col] == '-'
while not game_over:
    if current_player == 'X':
        print_board(board)
        row = int(input("Enter the row (0-2): "))
        col = int(input("Enter the col (0-2): "))
        while(not is_valid_move(board,row,col)):
            print("Ahh! invalid move")
            row = int(input("Enter the row (0-2): "))
            col = int(input("Enter the col (0-2): "))
        action = (row, col)
    else:
        action = choose_action(board,exploration_rate=0.1)
    row,col = action
    board[row,col] = current_player

    game_over, winner = is_game_over(board)

    if game_over:
        print_board(board)
        if winner == 'X':
            print("You win")
        elif winner == 'O':
            print("agent wins")
        else:
            print("Draw")

    else:
        current_player = players[(players.index(current_player) + 1) % num_players]
