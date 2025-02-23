import random
import pygame
import math
from connect4 import connect4
import sys
import numpy as np
import time

class connect4Player(object):
	def __init__(self, position, seed=0, CVDMode=False):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)
		if CVDMode:
			global P1COLOR
			global P2COLOR
			P1COLOR = (227, 60, 239)
			P2COLOR = (0, 255, 0)

	def play(self, env: connect4, move_dict: dict) -> None:
		move_dict["move"] = -1

class humanConsole(connect4Player):
	'''
	Human player where input is collected from the console
	'''
	def play(self, env: connect4, move_dict: dict) -> None:
		move_dict['move'] = int(input('Select next move: '))
		while True:
			if int(move_dict['move']) >= 0 and int(move_dict['move']) <= 6 and env.topPosition[int(move_dict['move'])] >= 0:
				break
			move_dict['move'] = int(input('Index invalid. Select next move: '))

class humanGUI(connect4Player):
	'''
	Human player where input is collected from the GUI
	'''

	def play(self, env: connect4, move_dict: dict) -> None:
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, P1COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
					else: 
						pygame.draw.circle(screen, P2COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move_dict['move'] = col
					done = True

class randomAI(connect4Player):
	'''
	connect4Player that elects a random playable column as its move
	'''

	def play(self, env: connect4, move_dict: dict) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move_dict['move'] = random.choice(indices)

class stupidAI(connect4Player):
	'''
	connect4Player that will play the same strategy every time
	Tries to fill specific columns in a specific order 
	'''
	def play(self, env: connect4, move_dict: dict) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move_dict['move'] = 3
		elif 2 in indices:
			move_dict['move'] = 2
		elif 1 in indices:
			move_dict['move'] = 1
		elif 5 in indices:
			move_dict['move'] = 5
		elif 6 in indices:
			move_dict['move'] = 6
		else:
			move_dict['move'] = 0

class minimaxAI(connect4Player):
	'''
	This is where you will design a connect4Player that 
	implements the minimiax algorithm WITHOUT alpha-beta pruning
	'''
	def __init__(self, position, seed, cvd_mode, depth = 4):
		super().__init__(position, seed, cvd_mode)
		self.depth = depth
		self.opponent_position = 2 if position == 1 else 1
		self.cvd_mode = cvd_mode
	
	def evaluate(self, board):
		# Weights and scores initialization
		window_weights = [1, 10, 100, 10000]
		player_score = 0
		opponent_score = 0
		center_bonus = 0
		# Helper to evaluate a window
		def evaluate_window(window, player):
			score = 0
			if window.count(player) == 4:
				score += window_weights[3]
			elif window.count(player) == 3 and window.count(0) == 1:
				score += window_weights[2]
			elif window.count(player) == 2 and window.count(0) == 2:
				score += window_weights[1]
			return score

		# Sequence scoring
		for r in range(6):
			for c in range(7-3):
				# Horizontal window
				window = list(board[r, c:c+4])
				player_score += evaluate_window(window, self.position)
				opponent_score += evaluate_window(window, self.opponent_position)
				
				# Vertical window
				if r <= 2:
					window = list(board[r:r+4, c])
					player_score += evaluate_window(window, self.position)
					opponent_score += evaluate_window(window, self.opponent_position)

				# Diagonal windows
				if r <= 2 and c <= 3:
					window = [board[r+i][c+i] for i in range(4)]
					player_score += evaluate_window(window, self.position)
					opponent_score += evaluate_window(window, self.opponent_position)
				
				# Anti-diagonal windows
				if r >= 3 and c <= 3:
					window = [board[r-i][c+i] for i in range(4)]
					player_score += evaluate_window(window, self.position)
					opponent_score += evaluate_window(window, self.opponent_position)
		# Center column bonus
		center_columns = [2, 3, 4]  # Adjust based on your game's column indices
		for r in range(6):
			for c in center_columns:
				if board[r][c] == self.position:
					center_bonus += 4 if c == 3 else 2 if c in [2, 4] else 1

		return player_score - opponent_score + center_bonus
		

	
	def minimax(self, board, depth, maximizing_player):
		valid_moves = [c for c in range(7) if board[0][c] == 0]
		if depth == 0 or not valid_moves:
			return (None, self.evaluate(board))

		if maximizing_player:
			max_eval = -float('inf')
			best_col = np.random.choice(valid_moves)
			for col in valid_moves:
				row = next(r for r in range(6) if board[r][col] == 0)
				b_copy = np.copy(board)
				b_copy[row][col] = self.position
				new_score = self.minimax(b_copy, depth-1, False)[1]
				if new_score > max_eval:
					max_eval = new_score
					best_col = col
			return (best_col, max_eval)
		else:
			min_eval = float('inf')
			best_col = np.random.choice(valid_moves)
			for col in valid_moves:
				row = next(r for r in range(6) if board[r][col] == 0)
				b_copy = np.copy(board)
				b_copy[row][col] = self.opponent_position
				new_score = self.minimax(b_copy, depth-1, True)[1]
				if new_score < min_eval:
					min_eval = new_score
					best_col = col
			return (best_col, min_eval)
   
   
 

	def play(self, env, move_dict):
		board = np.array(env.getBoard())
		best_move, _ = self.minimax(board, self.depth, True)
		move_dict['move'] = best_move
	
class alphaBetaAI(connect4Player):
	def __init__(self, position, seed, cvd_mode,  depth=8):
		super().__init__(position, seed)
		self.initial_depth = depth
		self.cvd_mode = cvd_mode
		self.time_limit = 2.99  # 2.99s safety margin
		self.opponent_position = 2 if position == 1 else 1
		self.center_order = [3, 2, 4, 1, 5, 0, 6]  # Critical move ordering

	def play(self, env, move_dict):
		board = np.array(env.getBoard())
		best_move = self.iterative_deepening_search(board)
		move_dict['move'] = best_move

	def iterative_deepening_search(self, board):
		self.start_time = time.time()
		best_move = 3  # Default to center
		for depth in range(1, self.initial_depth + 1):
			if time.time() - self.start_time > 2.5:  # Leave 0.5s buffer
				break
			try:
				move, _ = self.alphabeta(board, depth, -math.inf, math.inf, True)
				best_move = move
			except TimeoutError:
				break
		return best_move

	def alphabeta(self, board, depth, alpha, beta, maximizing_player):
		if time.time() - self.start_time > self.time_limit:
			raise TimeoutError()

		# Fast terminal check
		terminal, score = self.is_terminal(board)
		if terminal or depth == 0:
			return (None, score)

		# Prioritize center columns first
		valid_moves = [c for c in self.center_order if board[0][c] == 0]

		best_col = valid_moves[0] if valid_moves else None
		best_val = -math.inf if maximizing_player else math.inf

		for col in valid_moves:
			row = next(r for r in range(5, -1, -1) if board[r][col] == 0)
			b_copy = np.copy(board)
			b_copy[row][col] = self.position if maximizing_player else self.opponent_position

			# Recursive search
			current_val = self.alphabeta(b_copy, depth-1, alpha, beta, not maximizing_player)[1]

			if maximizing_player:
				if current_val > best_val:
					best_val = current_val
					best_col = col
				alpha = max(alpha, best_val)
			else:
				if current_val < best_val:
					best_val = current_val
					best_col = col
				beta = min(beta, best_val)

			if alpha >= beta:
				break  # Prune

		return (best_col, best_val)

	def is_terminal(self, board):
		"""Ultra-fast terminal check optimized for speed"""
		# Check horizontal
		for r in range(6):
			for c in range(4):
				if board[r][c] == board[r][c+1] == board[r][c+2] == board[r][c+3] != 0:
					return (True, 10000 if board[r][c] == self.position else -10000)

		# Check vertical
		for c in range(7):
			for r in range(3):
				if board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c] != 0:
					return (True, 10000 if board[r][c] == self.position else -10000)

		# Check diagonals
		for r in range(3):
			for c in range(4):
				if board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3] != 0:
					return (True, 10000 if board[r][c] == self.position else -10000)
				if board[r+3][c] == board[r+2][c+1] == board[r+1][c+2] == board[r][c+3] != 0:
					return (True, 10000 if board[r+3][c] == self.position else -10000)

		# Check draw
		if np.all(board[0] != 0):
			return (True, 0)

		return (False, 0)
# Defining Constants
SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
P1COLOR = (255,0,0)
P2COLOR = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)



