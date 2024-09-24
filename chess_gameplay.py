import torch
import torch.nn as nn
import numpy as np
from random import choice, choices
from itertools import accumulate
import chess
import chess.svg
from chess import Board
from chess.engine import SimpleEngine, Limit
import cairosvg
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import math

WHEREAMI = os.path.dirname(__file__)
STOCKFISH_PATH = os.path.join(WHEREAMI, 'utils', 'stockfish')

def softmax_temp(x, temp=1):
    z = np.exp((x - x.max()) / temp)
    return z / z.sum()

def entropy(d):
    # Returns the entropy of a discrete distribution
    e = -(d * np.log2(d + 1e-10)).sum() # epsilon value added due to log2(0) == undefined.
    return e

def entropy_temperature(x, target_entropy, T=[1e-3, 1e0, 1e2], tol=1e-3, max_iter=10_000):
    # returns the temperature parameter (to within tol) required to transform the vector x into a 
    # probability distribution with a particular target entropy
    delta = np.inf
    for i in range(max_iter):
        if delta > tol:
            E = [entropy(softmax_temp(x, temp=t)) for t in T]
            if E[0] > target_entropy:
                T = [T[0]/2, T[1], T[2]]
            elif E[2] < target_entropy:
                T = [T[0], T[1], T[2]*2]
            elif E[0] < target_entropy < E[1]:
                T = [T[0], (T[0]+T[1])/2, T[1]]
            elif E[1] < target_entropy < E[2]:
                T = [T[1], (T[1]+T[2])/2, T[2]]
            delta = (E[2] - E[0]) / target_entropy
        else:
            return (T[0]+T[2]) / 2
    print("WARNING: Entropy search depth exceeded.")
    return (T[0]+T[2]) / 2

def sans_to_pgn(move_sans):
    pgn = ["1."]
    for i,san in enumerate(move_sans, start=1):
        pgn += [san, " "]
        if i % 2 == 0:
            pgn.append(f"{int((i+2)/2)}.")
    return ''.join(pgn)

def selector(scores, p=0.3, k=3):
    '''
    Squashes the options distribution to have a target (lower) entropy.
    Selects a token, based on log2(p * len(k)) degrees of freedom.
    '''
    # If there is no variance in the scores, then just chose randomly.
    if all([score == scores[0] for score in scores]): 
        return choice(range(len(scores)))
    else:
        # Otherwise target entropy is either proportion p * max_possible_entropy (for small option sets) or 
        # as-if k-degree of freedom distribution (for num_scores >> k)
        target_entropy = min(p * np.log2(len(scores)), np.log2(k))
        # If we abandon the second term above, we allow the model more freedom when there are more options to 
        # chose from. Actually we could achieve the same thing by setting k ~ inf. Numpy handles this just fine 
        # so np.log2(float('inf')) = inf
        t = entropy_temperature(scores, target_entropy)
        dist = softmax_temp(scores, temp=t)
        return choices(range(len(scores)), cum_weights=list(accumulate(dist)))[0]

class Agent:
    def __init__(self, model=None, p=0.3, k=3):
        self.model, self.p, self.k = model, p, k

        if self.model:
            assert isinstance(model, nn.Module), "ERROR: model must be a torch nn.Module"
            self.model.eval()

    def select_move(self, pgn, legal_moves):
        # If there is no model passed, then just chose randomly.
        if self.model is None:
            return choice(legal_moves)

        scores = []
        with torch.no_grad():
            for move in legal_moves:
                score = self.model.score(pgn, move)
                scores.append(score)

        # Index of selected move
        selection = selector(np.array(scores), self.p, self.k)
        return legal_moves[selection]

def play_game(agents, teams, max_moves=float('inf'), min_seconds_per_move=2, verbose=False, poseval=False, image_path="/mnt/chess/", eval_time_limit=2, eval_depth_limit=25):

    board = Board()
    if poseval:
        white_score = (evaluate_position(board, time_limit=2, depth_limit=25) + 10_000) / 20_000
    else:
        white_score = 0.5

    move_sans = [] # for constructing the pgn

    if image_path:
        render_game_board(board, teams, white_score=0.5, winner=None, out_path=image_path)
    game_result = {'white': {'moves': [], 'points': 0}, 'black': {'moves': [], 'points': 0}, 'all_moves': [(board, None)]}

    # Play a game until game over.
    while True:

        start = time.perf_counter()
        whites_turn = board.turn
        turn = "white" if whites_turn else "black"
        
        # Check if checkmate or draw.
        player_points, opponent_points = (None, None)

        checkmate = board.is_checkmate()
        draw = board.is_variant_draw()
        stalemate = board.is_stalemate()

        if checkmate:
            player_points, opponent_points = (-1.0, 1.0)
            if verbose:
                winner = "white" if turn == "black" else "black"
                print(f"Checkmate! {winner} wins!")

        elif draw or stalemate:
            player_points, opponent_points = (0.0, 0.0)
            if verbose:
                print("Draw or Stalemate.")

        elif len(game_result[turn]['moves']) >= max_moves:
            if poseval:
                score = evaluate_position(board, time_limit=eval_time_limit, depth_limit=eval_depth_limit)
                player_points, opponent_points = (score, -score)
            else:
                player_points, opponent_points = (0.0, 0.0)
            if verbose:
                print("Max moves reached.")

        if player_points is not None:
            player, opponent = ('white', 'black') if whites_turn else ('black','white')
            game_result[player]['points'] = player_points
            game_result[opponent]['points'] = opponent_points
            if verbose:
                white_points, black_points = game_result['white']['points'], game_result['black']['points']
                white_score, black_score = (white_points + 10_000) / 20_000, (black_points + 10_000) / 20_000
                print(f"White score: {white_score:,.3}, Black score: {black_score:,.3}")
            return game_result

        # generate legal move sans
        legal_moves = list(board.legal_moves)
        legal_move_sans = [board.san(move) for move in legal_moves]

        # agent selects move
        pgn = sans_to_pgn(move_sans)
        selected_move_san = agents[turn].select_move(pgn, legal_move_sans)
        selected_move = legal_moves[legal_move_sans.index(selected_move_san)]
        move_sans.append(selected_move_san)

        # push move to the board
        board.push_san(selected_move_san)

        # evaluate the board:
        if poseval:
            score = evaluate_position(board, time_limit=eval_time_limit, depth_limit=eval_depth_limit)
            # if white just moved, then it's now black's turn, so the score is black's score
            if turn == 'white':
                white_score = (- score + 10_000) / 20_000
            else:
                white_score = (score + 10_000) / 20_000

        if image_path:
            render_game_board(board, teams, white_score=white_score, last_move=selected_move, winner=None, out_path=image_path)

        # Add this move to the game_record
        game_result[turn]['moves'].append((board, selected_move_san))
        game_result['all_moves'].append((board, selected_move_san))

        if verbose:
            print(f"{turn}: {selected_move_san}")

        # Delay next move so that humans can watch!
        move_duration = time.perf_counter() - start
        time_remaining = min_seconds_per_move - move_duration
        if time_remaining > 0:
            time.sleep(time_remaining)

def render_game_board(board, teams, white_score, last_move=None, winner=None, out_path=None):

    team_white = teams["white"]
    team_black = teams["black"]

    if winner == team_white:
        team_white += " ♛♚"
    elif winner == team_black:
        team_black += " ♛♚"
        
    board_svg = chess.svg.board(board, size=1000, orientation=chess.WHITE, lastmove=last_move, borders=False, coordinates=True, colors={"margin": "black"})
    board_png = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))

    fig = plt.figure(figsize=(10, 11.5))
    fig.set_facecolor('black')
    height, width = (50, 50)
    grid = (width, height)
    font_size = 50

    banner_depth = 0.1
    banner_depth_panels = math.floor(banner_depth * height)

    eval_bar_width = 0.08
    eval_bar_width_panels = math.floor(eval_bar_width * width)
    eval_bar_depth_panels = height - 2 * banner_depth_panels

    board_width_panels = width - eval_bar_width_panels

    # First row with one axis
    ax1 = plt.subplot2grid(grid, (0, 0), rowspan=banner_depth_panels, colspan=width)
    ax1.set_facecolor('black')
    ax1.text(0.04, 0.68, team_black, color='white', fontsize=font_size, ha='left', va='top', fontweight='bold')

    # Second row with three axes arranged horizontally
    ax2 = plt.subplot2grid(grid, (banner_depth_panels, 0), rowspan=eval_bar_depth_panels, colspan=eval_bar_width_panels)
    ax2.set_facecolor('black')
    ax2.axvspan(xmin=-0.03, xmax=0.04, color='white', alpha=1, ymax=white_score)
    ax2.axhline(y=0.5, color='red', linestyle='-', linewidth=4)  # Add a dashed horizontal white line at score 0

    # The board
    ax3 = plt.subplot2grid(grid, (banner_depth_panels, eval_bar_width_panels), rowspan=eval_bar_depth_panels, colspan=board_width_panels)
    img = Image.open(BytesIO(board_png))
    ax3.imshow(img)

    # Third row with one axis
    ax5 = plt.subplot2grid(grid, (height - banner_depth_panels, 0), rowspan=banner_depth_panels, colspan=width)
    ax5.text(0.04, 0.85, team_white, color='white', fontsize=font_size, ha='left', va='top', fontweight='bold')
    ax5.set_facecolor('black')

    for ax in [ax1, ax2, ax3, ax5]:
        ax.set_title('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)
        ax.axis('off')

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def evaluate_position(board, time_limit=2, depth_limit=25, STOCKFISH_PATH=STOCKFISH_PATH):
    # Initialize the Stockfish engine
    with SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        # Perform the evaluation
        info = engine.analyse(board, Limit(depth=depth_limit, time=time_limit))
        # Extract the score
        score = info['score'].relative.score(mate_score=10_000)
    return score