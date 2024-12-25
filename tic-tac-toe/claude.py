import random
import numpy as np
from collections import defaultdict

class TicTacToeRL:
    def __init__(self):
        self.board_size = 3
        self.players = ['X', 'O']
        self.learning_rate = 0.2  # Increased for faster learning
        self.discount_factor = 0.99  # Higher discount to value future rewards more
        self.exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.9999  # Slower decay for better exploration
        self.Q = defaultdict(lambda: np.zeros((self.board_size, self.board_size)) - 1)  # Initialize with negative values

    def reset_board(self):
        return np.array([['-' for _ in range(self.board_size)] for _ in range(self.board_size)])

    def board_to_string(self, board):
        return ''.join(board.flatten())

    def print_board(self, board):
        for row in board:
            print(' | '.join(row))
            print('-' * 9)

    def get_valid_moves(self, board):
        return [(i, j) for i in range(self.board_size) 
            for j in range(self.board_size) if board[i][j] == '-']

    def check_winning_move(self, board, player, move):
        """Check if a move would result in a win"""
        temp_board = board.copy()
        temp_board[move[0]][move[1]] = player
        game_over, winner = self.is_game_over(temp_board)
        return game_over and winner == player

    def check_blocking_move(self, board, player, move):
        """Check if a move would block opponent's win"""
        opponent = 'X' if player == 'O' else 'O'
        temp_board = board.copy()
        temp_board[move[0]][move[1]] = opponent
        game_over, winner = self.is_game_over(temp_board)
        return game_over and winner == opponent

    def is_game_over(self, board):
        # Check rows and columns
        for i in range(self.board_size):
            if len(set(board[i])) == 1 and board[i][0] != '-':
                return True, board[i][0]
            if len(set(board[:, i])) == 1 and board[0][i] != '-':
                return True, board[0][i]

        # Check diagonals
        diag = [board[i][i] for i in range(self.board_size)]
        anti_diag = [board[i][self.board_size-1-i] for i in range(self.board_size)]

        if len(set(diag)) == 1 and diag[0] != '-':
            return True, diag[0]
        if len(set(anti_diag)) == 1 and anti_diag[0] != '-':
            return True, anti_diag[0]

        # Check for draw
        if '-' not in board:
            return True, 'draw'

        return False, None

    def evaluate_board(self, board, player):
        """Evaluate the current board state"""
        game_over, winner = self.is_game_over(board)
        if game_over:
            if winner == player:
                return 1
            elif winner == 'draw':
                return 0.5
            else:
                return -1
        return 0

    def choose_action(self, board, current_player, training=True):
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return None

        # During actual gameplay or with low exploration rate, use strategic moves
        if not training or random.random() > self.exploration_rate:
            # First check for winning moves
            for move in valid_moves:
                if self.check_winning_move(board, current_player, move):
                    return move

            # Then check for blocking moves
            for move in valid_moves:
                if self.check_blocking_move(board, current_player, move):
                    return move

            # If center is available, take it
            if (1, 1) in valid_moves:
                return (1, 1)

            # If no strategic moves, use Q-values with some randomization
            state = self.board_to_string(board)
            q_values = [(move, self.Q[state][move]) for move in valid_moves]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_moves = [move for move, q in q_values if abs(q - max_q) < 1e-10]
            return random.choice(best_moves)

        # Exploration: choose random move
        return random.choice(valid_moves)

    def update_q_value(self, state, action, next_board, reward, current_player):
        """Update Q-value with enhanced rewards"""
        next_state = self.board_to_string(next_board)
        next_max = np.max(self.Q[next_state])

        # Modify reward based on position control and strategic value
        if action == (1, 1):  # Center position
            reward += 0.1
        elif action in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # Corners
            reward += 0.05

        # Consider defensive moves
        if self.check_blocking_move(next_board, current_player, action):
            reward += 0.2

        self.Q[state][action] += self.learning_rate * (
            reward + self.discount_factor * next_max - self.Q[state][action]
        )

    def train(self, episodes):
        wins = 0
        draws = 0

        for episode in range(episodes):
            board = self.reset_board()
            game_over = False
            current_player = random.choice(self.players)
            moves_history = []

            while not game_over:
                state = self.board_to_string(board)
                action = self.choose_action(board, current_player, training=True)

                if action is None:
                    break

                row, col = action
                board[row][col] = current_player
                moves_history.append((state, action, current_player))

                game_over, winner = self.is_game_over(board)

                # Calculate reward
                if game_over:
                    if winner == current_player:
                        reward = 1
                        if current_player == 'O':
                            wins += 1
                    elif winner == 'draw':
                        reward = 0.8  # Higher reward for draws
                        draws += 1
                    else:
                        reward = -1

                    # Update Q-values for all moves in the game
                    for state, action, player in moves_history:
                        self.update_q_value(state, action, board, reward, player)
                        reward *= -1  # Alternate reward for opponent's moves

                current_player = 'O' if current_player == 'X' else 'X'

            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )

            if (episode + 1) % 1000 == 0:
                win_rate = wins / 1000
                draw_rate = draws / 1000
                print(f"Episode {episode + 1}/{episodes}")
                print(f"Win Rate: {win_rate:.2f}")
                print(f"Draw Rate: {draw_rate:.2f}")
                print(f"Exploration Rate: {self.exploration_rate:.4f}")
                print("-" * 40)
                wins = 0
                draws = 0

    def play_game(self):
        board = self.reset_board()
        current_player = 'X'  # Human always starts
        game_over = False

        while not game_over:
            self.print_board(board)

            if current_player == 'X':
                while True:
                    try:
                        row = int(input("Enter row (0-2): "))
                        col = int(input("Enter column (0-2): "))
                        if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == '-':
                            break
                        print("Invalid move! Try again.")
                    except ValueError:
                        print("Please enter numbers between 0 and 2.")
            else:
                print("AI is thinking...")
                action = self.choose_action(board, current_player, training=False)
                if action is None:
                    break
                row, col = action

            board[row][col] = current_player
            game_over, winner = self.is_game_over(board)

            if game_over:
                self.print_board(board)
                if winner == 'draw':
                    print("It's a draw!")
                else:
                    print(f"{'You' if winner == 'X' else 'AI'} win!")
            else:
                current_player = 'O' if current_player == 'X' else 'X'

# Usage
if __name__ == "__main__":
    agent = TicTacToeRL()
    print("Training the agent...")
    agent.train(100000)  # Increased training episodes

    while True:
        agent.play_game()
        play_again = input("Play again? (y/n): ").lower()
        if play_again != 'y':
            break
