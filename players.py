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
	
class TranspositionTable:
	class EntryType:
		EXACT = 0
		LOWER = 1
		UPPER = 2

	def __init__(self, size=1048583):  # Use a large prime number
		self.size = size
		self.table = [None] * size

	def put(self, key, score, flag, depth):
		index = key % self.size
		self.table[index] = (key, score, flag, depth)

	def get(self, key):
		index = key % self.size
		entry = self.table[index]
		if entry and entry[0] == key:
			return entry
		return None

class alphaBetaAI(connect4Player):
	def __init__(self, position, seed, cvd_mode, depth=5):  # Increased depth
		super().__init__(position, seed)
		self.initial_depth = depth
		self.time_limit = 1.99  # 2s with safety margin
		self.cvd_mode = cvd_mode
		self.opponent_position = 2 if position == 1 else 1
		self.center_order = [3, 2, 4, 1, 5, 0, 6]  # Center-first move order
		self.trans_table = TranspositionTable()
	
	def board_to_key(self, board):
		# Convert board to hashable key (simple version)
		return hash(tuple(map(tuple, board)))

	def evaluate(self, board):
		
		# Precompute next available rows for all columns
		next_available_rows = {}
		for c in range(7):
			next_row = -1
			for r in range(5, -1, -1):
				if board[r][c] == 0:
					next_row = r
					break
			next_available_rows[c] = next_row
		
		def count_sequences(player):
			consecutive_open_twos = 0
			potential_fours = 0
			open_threes = 0
			fours = 0

			# Horizontal checks
			for r in range(6):
				row = board[r]
				for c in range(4):
					# Check consecutive two patterns
					if row[c] == row[c+1] == player and row[c+2] == 0 and row[c+3] == 0:
						consecutive_open_twos += 1
					# Check for three-with-gap patterns (xx_x)
					if (row[c] == row[c+1] == row[c+3] == player and row[c+2] == 0) or \
					(row[c] == row[c+2] == row[c+3] == player and row[c+1] == 0):
						potential_fours += 1

			# Vertical checks
			for c in range(7):
				col = board[:,c]
				for r in range(3):
					if col[r] == col[r+1] == player and col[r+2] == 0 and col[r+3] == 0:
						consecutive_open_twos += 1

			# Diagonal checks (both directions)
			for r in range(3):
				for c in range(4):
					# Negative slope
					if board[r][c] == board[r+1][c+1] == player and board[r+2][c+2] == 0 and board[r+3][c+3] == 0:
						consecutive_open_twos += 1
					# Positive slope
					if board[r+3][c] == board[r+2][c+1] == player and board[r+1][c+2] == 0 and board[r][c+3] == 0:
						consecutive_open_twos += 1
			
			

			return {
				'consecutive_open_twos': consecutive_open_twos,
				'potential_fours': potential_fours,
				'open_threes': open_threes,
				'fours': fours
			}

		my_stats = count_sequences(self.position)
		opp_stats = count_sequences(self.opponent_position)
		
		# Strategic weighting with threat prioritization
		utility = (
			(my_stats['fours'] * 10000) +
			(my_stats['potential_fours'] * 5000) +
			(my_stats['open_threes'] * 200) +
			(my_stats['consecutive_open_twos'] * 50) -  # Increased weight
			(opp_stats['fours'] * 10000) -
			(opp_stats['potential_fours'] * 6000) -     # Higher penalty
			(opp_stats['open_threes'] * 300) -
			(opp_stats['consecutive_open_twos'] * 100)  # Heavy penalty for opponent's threats
		)
		
		# Add center control
		center_value = sum(board[r][c] == self.position 
						for c in [2,3,4] for r in range(6))
		return utility + center_value * 3

	def get_valid_moves_ordered(self, board, critical_col=None):
		"""Prioritize critical blocking moves first"""
		valid = [c for c in self.center_order if board[0][c] == 0]
		if critical_col is not None and critical_col in valid:
			valid.remove(critical_col)
			return [critical_col] + valid
		return valid

	def iterative_deepening_search(self, board):
		self.start_time = time.time()
		best_move = 3  # Default to center
		depth = 1
		
		while depth <= self.initial_depth:
			try:
				move, _ = self.alphabeta(board, depth, -float('inf'), float('inf'), True)
				best_move = move
				depth += 1
			except TimeoutError:
				break
		return best_move

	def alphabeta(self, board, depth, alpha, beta, maximizing_player):
		if time.time() - self.start_time > self.time_limit:
			raise TimeoutError()
		
		original_alpha = alpha
		key = self.board_to_key(board)
		entry = self.trans_table.get(key)

		if entry:
			entry_key, entry_score, entry_flag, entry_depth = entry
			if entry_depth >= depth:
				if entry_flag == TranspositionTable.EntryType.EXACT:
					return (None, entry_score)
				elif entry_flag == TranspositionTable.EntryType.LOWER:
					alpha = max(alpha, entry_score)
				elif entry_flag == TranspositionTable.EntryType.UPPER:
					beta = min(beta, entry_score)
				
				if alpha >= beta:
					return (None, entry_score)

		terminal, value = self.is_terminal(board)
		if depth == 0 or terminal:
			return (None, value)

		critical_col = self.find_urgent_block(board)
		valid_moves = self.get_valid_moves_ordered(board, critical_col)
		best_col = valid_moves[0] if valid_moves else None

		if maximizing_player:
			max_val = -float('inf')
			for col in valid_moves:
				row = next(r for r in range(5, -1, -1) if board[r][col] == 0)
				b_copy = np.copy(board)
				b_copy[row][col] = self.position
				current_val = self.alphabeta(b_copy, depth-1, alpha, beta, False)[1]
				
				if current_val > max_val:
					max_val = current_val
					best_col = col
				alpha = max(alpha, max_val)
				if alpha >= beta:
					break
			
			# Store in transposition table
			if max_val <= original_alpha:
				flag = TranspositionTable.EntryType.UPPER
			elif max_val >= beta:
				flag = TranspositionTable.EntryType.LOWER
			else:
				flag = TranspositionTable.EntryType.EXACT
			self.trans_table.put(key, max_val, flag, depth)
			
			return (best_col, max_val)
		else:
			min_val = float('inf')
			for col in valid_moves:
				row = next(r for r in range(5, -1, -1) if board[r][col] == 0)
				b_copy = np.copy(board)
				b_copy[row][col] = self.opponent_position
				current_val = self.alphabeta(b_copy, depth-1, alpha, beta, True)[1]
				
				if current_val < min_val:
					min_val = current_val
					best_col = col
				beta = min(beta, min_val)
				if alpha >= beta:
					break
			
			# Store in transposition table
			if min_val <= original_alpha:
				flag = TranspositionTable.EntryType.UPPER
			elif min_val >= beta:
				flag = TranspositionTable.EntryType.LOWER
			else:
				flag = TranspositionTable.EntryType.EXACT
			self.trans_table.put(key, min_val, flag, depth)
			
			return (best_col, min_val)

	def is_terminal(self, board):
		# Fast win check using bitwise operations
		for player in [self.position, self.opponent_position]:
			# Horizontal
			for r in range(6):
				for c in range(4):
					if all(board[r][c+i] == player for i in range(4)):
						return (True, 100000 if player == self.position else -100000)
			# Vertical
			for c in range(7):
				for r in range(3):
					if all(board[r+i][c] == player for i in range(4)):
						return (True, 100000 if player == self.position else -100000)
			# Diagonals
			for r in range(3):
				for c in range(4):
					if all(board[r+i][c+i] == player for i in range(4)):
						return (True, 100000 if player == self.position else -100000)
					if all(board[r+3-i][c+i] == player for i in range(4)):
						return (True, 100000 if player == self.position else -100000)
		# Check tie
		if all(board[0][c] != 0 for c in range(7)):
			return (True, 0)
		return (False, 0)
	
	def find_urgent_block(self, board):
		"""Quick check for immediate opponent threats needing block"""
		for c in self.center_order:
			if board[0][c] != 0:
				continue
			# Simulate opponent move
			row = next(r for r in range(5, -1, -1) if board[r][c] == 0)
			temp_board = np.copy(board)
			temp_board[row][c] = self.opponent_position
			if self.is_terminal(temp_board)[0]:
				return c
		return None
    
	def play(self, env, move_dict):
		board = np.array(env.getBoard())
		try:
			best_move = self.iterative_deepening_search(board)
		except TimeoutError:
			valid = self.get_valid_moves_ordered(board)
			best_move = valid[0] if valid else 3
		move_dict['move'] = best_move

    # Keep existing is_terminal and play methods
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




