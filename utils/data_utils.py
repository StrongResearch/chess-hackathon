import numpy as np
import requests
import tarfile
import zipfile
import os
import io
import chess.pgn
from chess import Board
from chess.engine import SimpleEngine, Limit
from h5py import File as h5pyFile
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from .constants import LCZERO_TEST60_URL, PGN_CHARS, PIECE_CHARS, STOCKFISH_PATH

## -- Generating Dataset of Leela Chess Zero PGNs in HDF format -- ##

def scrape_tar_bz2_links(url=LCZERO_TEST60_URL):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page: {url}")
    soup = BeautifulSoup(response.content, 'html.parser')
    a_tags = soup.find_all('a')
    tar_bz2_links = [f"{url}/{a['href']}" for a in a_tags if 'href' in a.attrs and a['href'].endswith('.tar.bz2')]
    return tar_bz2_links

def download_tar_files(urls, dest_dir):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest_dir_contents = os.listdir(dest_dir)
    missing_file_urls = [u for u in urls if u.split('/')[-1] not in dest_dir_contents]
    for url in tqdm(missing_file_urls, desc=f"Downloading {len(missing_file_urls)} PGN files"):
        filename = url.split('/')[-1]
        response = requests.get(url)
        with open(os.path.join(dest_dir, filename), 'wb') as file:
            file.write(response.content)

def save_pgn_batch_to_hdf(pgn_batch, hdf_count, dest_dir, metas=None):
    pgns_array = np.array(pgn_batch, dtype='S')
    hdf_name = f'pgnHDF{hdf_count}.h5'
    hdf_path = os.path.join(dest_dir, hdf_name)
    with h5pyFile(hdf_path, 'w') as hf:
        hf.create_dataset("pgn", data=pgns_array, compression='gzip', compression_opts=9)
        if metas:
            hf.create_dataset("meta", data=metas, compression='gzip', compression_opts=9)
    return hdf_name

def write_inventory(dest_dir, hdf_sizes, hdf_names):
    inventory_path = os.path.join(dest_dir, "inventory.txt")
    with open(inventory_path, "w") as file:
        file.write(f"Total pgns: {sum(hdf_sizes):,}\n")
        for size, name in zip(hdf_sizes, hdf_names):
            file.write(f"{size} {name}\n")

def compile_tars_to_hdfs(source_dir, dest_dir, batch_size=1_000_000):
    assert os.path.exists(source_dir), "ERROR: source_dir not found."
    assert not os.path.exists(dest_dir), "ERROR: dest_dir present, please delete first."
    Path(dest_dir).mkdir(parents=True, exist_ok=False)

    tar_files = [f for f in os.listdir(source_dir) if f.endswith('.tar.bz2')]
    pgn_batch = []
    hdf_count = 0
    hdf_sizes = []
    hdf_names = []

    for tfile in tqdm(tar_files, desc = "Processing tars into HDFs"):
        with tarfile.open(os.path.join(source_dir, tfile), "r:bz2") as tar:
            pgn_files = [file.name for file in tar.getmembers() if file.name.endswith(".pgn")]

            for pgnfile in pgn_files:
                pgn = tar.extractfile(tar.getmember(pgnfile)).read()
                try:
                    pgn = pgn.decode().strip()
                    assert set(pgn).issubset(set(PGN_CHARS))
                    pgn_batch.append(pgn)

                    if len(pgn_batch) == batch_size:
                        hdf_name = save_pgn_batch_to_hdf(pgn_batch, hdf_count, dest_dir)
                        hdf_sizes.append(batch_size)
                        hdf_names.append(hdf_name)
                        hdf_count += 1
                        pgn_batch = []
                except:
                    print(f"FAILED: {pgn}")
                    continue

    # Store leftover pgns in new hdf
    hdf_name = save_pgn_batch_to_hdf(pgn_batch, hdf_count, dest_dir)
    hdf_sizes.append(len(pgn_batch))
    hdf_names.append(hdf_name)
    hdf_count += 1
    pgn_batch = []

    write_inventory(dest_dir, hdf_sizes, hdf_names)

'''
# Example usage:

urls = scrape_tar_bz2_links(LCZERO_TEST60_URL)[0:50]
download_tar_files(urls)
compile_tars_to_hdfs(source_dir, dest_dir, batch_size=1_000_000)
'''


## -- Generating Dataset of Historic Grand Master Game PGNS in HDF format -- ##

def download_gm_pgns(url):
    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        name = list(z.namelist())[0]
        contents = z.read(name)
    splits = contents.split(b"\r\n\r\n")
    metas = list(map(lambda p: p.replace(b"\r", b""), splits[0::2]))
    pgns = list(map(lambda p: p.replace(b"\r\n", b" "), splits[1::2]))

    val_metas, val_pgns = [], []
    for meta, pgn in zip(metas, pgns):
        try:
            assert set(pgn.decode()).issubset(set(PGN_CHARS))
            val_metas.append(meta)
            val_pgns.append(pgn.decode())
        except:
            unrec_chars = set(pgn.decode()) - set(PGN_CHARS)
            print(f"ERROR: PGN has unrecognised chars: {unrec_chars}")
    assert len(val_metas) == len(val_pgns), "ERROR: len(metas) != len(pgns)"
    return val_metas, val_pgns

'''
# 10 Best players of all time
# https://www.chess.com/article/view/best-chess-players-of-all-time
urls = []
# 1. Garry Kasparov
urls.append("https://www.pgnmentor.com/players/Kasparov.zip")
# 2. Magnus Carlsen
urls.append("https://www.pgnmentor.com/players/Carlsen.zip")
# 3. Bobby Fischer
urls.append("https://www.pgnmentor.com/players/Fischer.zip")
# 4. José Raúl Capablanca
urls.append("https://www.pgnmentor.com/players/Capablanca.zip")
# 5. Anatoly Karpov
urls.append("https://www.pgnmentor.com/players/Karpov.zip")
# 6. Mikhail Botvinnik
urls.append("https://www.pgnmentor.com/players/Botvinnik.zip")
# 7. Vladimir Kramnik
urls.append("https://www.pgnmentor.com/players/Kramnik.zip")
# 8. Emanuel Lasker
urls.append("https://www.pgnmentor.com/players/Lasker.zip")
# 9. Mikhail Tal
urls.append("https://www.pgnmentor.com/players/Tal.zip")
# 10. Alexander Alekhine
urls.append("https://www.pgnmentor.com/players/Alekhine.zip")

metas, pgns = [], []
for url in tqdm(urls):
    m, p = download_gm_pgns(url)
    metas += m
    pgns += p

save_pgn_batch_to_hdf(pgns, hdf_count=0, dest_dir)
'''

## -- Generating Dataset of Board Evaluations from HDF dataset -- ##

def encode_board(board, symmetric=True):
    # IF SYMMETRIC:
    # If board.turn = 1 then it is now white's turn which means this is a potential move
    # being contemplated by black, and therefore we flip the board to black's perspective.
    # for black's perspective
    # If board.turn = 0 then it is now black's turn which means this is a potential move
    # being contemplated by white, and therefore we leave the board the way it is.
    step = 1 - 2 * board.turn if symmetric else 1
    unicode = board.unicode().replace(' ','').replace('\n','')[::step]
    return np.array([PIECE_CHARS[::step].index(c) for c in unicode], dtype=int).reshape(8,8)

def score_possible_boards(board, engine, depth_limt=15, time_limit=2, topk=None):
    legal_moves = board.legal_moves
    legal_move_sans = [board.san(move) for move in legal_moves]
    encoded_boards = []
    scores = []
    for move in legal_moves:
        possible_board = board.copy()
        possible_board.push(move)
        possible_board_encoded = encode_board(possible_board)
        info = engine.analyse(possible_board, Limit(depth=depth_limt, time=time_limit))
        score = info['score'].relative.score(mate_score=10_000)
        scores.append(score)
        encoded_boards.append(possible_board_encoded)
    # Negative of scores to represent from perspective of player to move.
    scores = [-s for s in scores]
    if topk:
        moves_sans_boards_scores = sorted(zip(legal_moves, legal_move_sans, encoded_boards, scores), key=lambda t: t[-1], reverse=True)
        legal_moves, legal_move_sans, encoded_boards, scores = zip(*moves_sans_boards_scores[:topk])
    return legal_moves, legal_move_sans, encoded_boards, scores

def pgn_to_board_evaluations(pgn, depth_limt=15, time_limit=2, topk=None, verbose=False):
    '''
    Accepts pgn in the following format (or a format that can be parsed as follows):
    "1.Nc3 d5 2.d4 c6 3.a3 Nf6 4.Bf4 Bf5 5.Nf3 Nh5 6.h4 Nxf4 7.h5 0-1 {OL: 0}"
    Returns array of scored boards (N, 8, 8) and board scores (N,)
    '''
    engine = SimpleEngine.popen_uci(STOCKFISH_PATH)
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = Board()
    scored_boards = []
    board_scores = []
    _legal_moves, _legal_move_sans, possible_boards, scores = score_possible_boards(board, engine, depth_limt=depth_limt, time_limit=time_limit, topk=topk)
    scored_boards += possible_boards
    board_scores += scores
    for move in (tqdm(list(game.mainline_moves())) if verbose else list(game.mainline_moves())):
        board.push(move)
        _legal_moves, _legal_move_sans, possible_boards, scores = score_possible_boards(board, engine, depth_limt=depth_limt, time_limit=time_limit, topk=topk)
        scored_boards += possible_boards
        board_scores += scores
    return np.array(scored_boards), np.array(board_scores)

'''
# Example usage:

pgn_dataset = PGN_HDF_Dataset(SOURCE_DIR)
pgn = pgn_dataset[0] # EXAMPLE FOR JUST ONE PGN
boards, scores = pgn_to_board_evaluations(pgn, engine, depth_limit, time_limit)
'''
