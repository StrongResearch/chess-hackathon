import torch
import torch.distributed as dist
import os
import time
import socket
from multiprocessing import Queue, Pool, Process
import h5py
from cycling_utils import atomic_torch_save
from utils.data_utils import pgn_to_board_evaluations
from utils.datasets import PGN_HDF_Dataset

SOURCE_DIR = "/data"
DEST_DIR = "/root/chess-hackathon-3/data/lc0_board_evals"

def worker_function(idx, depth_limit=20, time_limit=5*60):
    start = time.perf_counter()
    pgn_dataset = PGN_HDF_Dataset(SOURCE_DIR)
    pgn = pgn_dataset[idx]
    boards, scores = pgn_to_board_evaluations(pgn, depth_limit, time_limit)
    atomic_torch_save({"idx": idx, "boards": boards, "scores": scores}, os.path.join(DEST_DIR, f"{idx}.pack"))
    print(f"Generated {len(scores)} board evaluations at {len(scores) / (time.perf_counter() - start):,.2f} boards per second.")

def writer_function():
    boards_hdf_path = os.path.join(DEST_DIR, "boards.h5")
    scores_hdf_path = os.path.join(DEST_DIR, "scores.h5")
    while True:
        packable_files = [f for f in os.listdir(DEST_DIR) if f.endswith(".pack")]
        if packable_files:
            file_path = os.path.join(DEST_DIR, packable_files[0])
            contents = torch.load(file_path)
            with h5py.File(boards_hdf_path, 'a') as boards_hdf:
                 boards_hdf.create_dataset(str(contents["idx"]), data=contents["boards"])
            with h5py.File(scores_hdf_path, 'a') as scores_hdf:
                scores_hdf.create_dataset(str(contents["idx"]), data=contents["scores"])
            os.remove(file_path)
        else:
            continue

if __name__ == "__main__":

    hostname = socket.gethostname() # What's the name of this machine?
    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank on local node
    torch.cuda.set_device(local_rank)  # Enables calling 'cuda'

    if rank == 0:
        print("Starting up...")

    # init objects representing progress at start
    pgn_dataset = PGN_HDF_Dataset(SOURCE_DIR)
    num_already_processed = torch.tensor(0, device='cuda')
    for_procesing = range(len(pgn_dataset))

    if rank == 0:
        print("Init dataset complete.")

    # clean up any untrustworthy files from last run
    if rank == 0:
        removable_files = [f for f in os.listdir(DEST_DIR) if f.endswith(".pack")]
        if removable_files:
            print(f"Removing old files: {removable_files}")
        for f in removable_files:
            file_path = os.path.join(DEST_DIR, f)
            os.remove(file_path)

    if rank == 0:
        print("Removed any stale leftovers.")
    
    # rank 0 works out how many have already been processed
    if rank == 0:
        already_processed = []
        boards_hdf_path = os.path.join(DEST_DIR, "boards.h5")
        scores_hdf_path = os.path.join(DEST_DIR, "scores.h5")
        if os.path.exists(boards_hdf_path) and os.path.exists(scores_hdf_path):
            with h5py.File(os.path.join(DEST_DIR, "boards.h5"), 'r') as b_hdf:
                b_hdf_keys = set(b_hdf.keys())
            with h5py.File(os.path.join(DEST_DIR, "scores.h5"), 'r') as s_hdf:
                s_hdf_keys = set(s_hdf.keys())
            already_processed = list(set.intersection(b_hdf_keys, s_hdf_keys))
            already_processed = [int(idx) for idx in already_processed]
        num_already_processed = torch.tensor(len(already_processed), device='cuda')

    if rank == 0:
        print(f"Rank 0 announcing {len(already_processed):,} already processed of {len(pgn_dataset):,} total.")

    # rank 0 broadcasts that number to the other ranks so they can all prepare to recieve the tensor of already processed indexes
    dist.broadcast(num_already_processed, src=0)
    print(f"Rank {rank} reports {num_already_processed.item():,} PGNs already processed.")

    if num_already_processed.item() > 0:
        # preparing tensor to receive already processed indexes
        already_processed_tensor = torch.zeros((num_already_processed.item(),), dtype=torch.int32, device='cuda')
        # rank 0 replaces that locally with the actual tensor of already processed indexes
        if rank == 0:
            already_processed_tensor = torch.tensor(already_processed, dtype=torch.int32, device=local_rank)
        # and broadcasts that to the other processes
        dist.broadcast(already_processed_tensor, src=0)
        print(f"Rank {rank} reports sum of already processed indices is {int(already_processed_tensor.sum().item())}.")
        already_processed = set(already_processed_tensor.tolist())
        # updating for_processing based on already_processed
        for_procesing = [idx for idx in range(len(pgn_dataset)) if idx not in already_processed]

    # Reporting on the local batch of PGNs to process
    local_batch = for_procesing[rank::world_size]
    print(f"Hostname: {hostname} assigned {len(local_batch):,} PGNs.")

    # Start writer process only on master rank
    if rank == 0:
        write_queue = Queue()
        writer = Process(target=writer_function)
        writer.start()
        print(f"Rank {rank} Started writer process.")

    with Pool() as pool:
        print(f"Rank {rank} Starting worker processes.")
        pool.map(worker_function, local_batch)