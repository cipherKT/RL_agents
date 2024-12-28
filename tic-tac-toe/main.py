class TicTacToe:
    def __init__(self):
        self.board = [" "]*9
    def display_board(self):
        for row in range(3):
            print(" | ".join(self.board[row*3: (row+1)*3]))
            if(row<2):
                print("-"*10)
    def is_winner(self, player):
        win_conditions = [
            [0,1,2], [3,4,5], [6,7,8], # for rows
            [0,3,6], [1,4,7], [2,5,8], # for cols
            [0,4,8], # primary diagonal
            [2,4,6] # secondary diagonal
        ]
        for condition in win_conditions:
            if all(self.board[position] == player for position in condition):
                return True
        return False

    def is_draw(self):
        return " " not in self.board
    def valid_moves(self):
        valid_positions = []
        for i in range(9):
            if self.board[i] == " ":
                valid_positions.append(i)
        return valid_positions
    def make_move(self, position, player):
        if self.board[position] == " ":
            self.board[position] = player
            return True
        return False
    def undo_move(self, position):
        self.board[position] = " "
    def is_game_over(self):
        return self.is_winner("X") or self.is_winner("O") or self.is_draw()

class MiniMax:
    def __init__(self, player):
        self.player = player
        self.opponent = "O" if player == "X" else "X"
    def minimax(self, game, depth, maximizing_player, alpha = -float('inf'), beta = float('inf')):
        if game.is_winner(self.player):
            return 10 - depth
        if game.is_winner(self.opponent):
            return depth - 10
        if game.is_draw():
            return 0
        if maximizing_player:
            max_eval = -float('inf')
            for move in game.valid_moves():
                game.make_move(move, self.player)
                eval = self.minimax(game, depth+1, False, alpha, beta)
                game.undo_move(move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in game.valid_moves():
                game.make_move(move, self.opponent)
                eval = self.minimax(game, depth+1, True, alpha, beta)
                game.undo_move(move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    def get_best_move(self, game):
        best_move = None
        best_value = -float('inf')

        for move in game.valid_moves():
            game.make_move(move, self.player)
            move_value = self.minimax(game, 0, False)
            game.undo_move(move)

            if move_value > best_value:
                best_value = move_value
                best_move = move
        return best_move




# main
if __name__ == '__main__':
    game = TicTacToe()
    agent = MiniMax("X")
    print("*"*10 + "welcome to tic tac toe" + "*"*10)
    human_player = "X"  # default value
    try:
        human_player = input("Choose your player (X or O): ").strip().upper()  # Fixed strip() method
        while human_player not in ["X", "O"]:
            print("!"*5 + "INVALID CHOICE" + "!"*5)
            human_player = input("Choose your player (X or O): ").strip().upper()  # Fixed strip() method
    except Exception as e:
        print("!"*5+"Error occurred:"+ str(e) +"!"*5)
        print("*"*10+"Setting default player as X"+"*"*10)
    
    current_player = "X"

    while not game.is_game_over():
        game.display_board()
        print()
        if current_player == human_player:
            try:
                move = int(input("Your move(1-9): ")) - 1
                while move not in game.valid_moves():
                    print("!"*5 + "INVALID MOVE" + "!"*5)
                    move = int(input("Your move(1-9): ")) - 1
            except ValueError:
                print("!"*5 + "Please enter a number between 1 and 9" + "!"*5)
                continue
        else:
            print("Agent is thinking ....")
            move = agent.get_best_move(game)
        
        game.make_move(move, current_player)
        
        if game.is_game_over():
            break
            
        current_player = "O" if current_player == "X" else "X"
    
    game.display_board()
    if game.is_winner("X"):
        print("*"*10 + "X wins" + "*"*10)
    elif game.is_winner("O"):
        print("*"*10 + "O wins" + "*"*10)
    else:
        print("*"*10 + "Draw" + "*"*10)

