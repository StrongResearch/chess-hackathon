from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import os
import socket
import yaml
import time
import math

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    AtomicDirectory,
    atomic_torch_save,
)

from utils.optimizers import Lamb
from utils.datasets import EVAL_HDF_Dataset
from model import Model

timer.report("Completed imports")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", help="model config path", type=Path, default="/root/chess-hackathon/model_config.yaml")
    parser.add_argument("--load-path", help="path to checkpoint.pt file to resume from", type=Path, default="/root/chess-hackathon/recover/checkpoint.pt")
    parser.add_argument("--bs", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.01)
    parser.add_argument("--ws", help="learning rate warm up steps", type=int, default=1000)
    parser.add_argument("--grad-accum", help="gradient accumulation steps", type=int, default=6)
    parser.add_argument("--save-steps", help="saving interval steps", type=int, default=50)
    parser.add_argument("--dataset-id", help="Dataset ID for the dataset", type=str, required=True)
    return parser

def logish_transform(data):
    '''Zero-symmetric log-transformation.'''
    return torch.sign(data) * torch.log1p(torch.abs(data))

def spearmans_rho(a, b):
    '''Spearman's rank correlation coefficient'''
    assert len(a) == len(b), "ERROR: Vectors must be of equal length"
    n = len(a)
    a_ranks = [sorted(a).index(i) for i in a]
    b_ranks = [sorted(b).index(j) for j in b]
    a_ranks_mean = sum(a_ranks) / n
    b_ranks_mean = sum(b_ranks) / n
    rank_covariance = sum([(a_rank - a_ranks_mean) * (b_rank - b_ranks_mean) for a_rank, b_rank in zip(a_ranks, b_ranks)]) / n
    a_ranks_sd = (sum([(a_rank - a_ranks_mean) ** 2 for a_rank in a_ranks]) / n) ** 0.5
    b_ranks_sd = (sum([(b_rank - b_ranks_mean) ** 2 for b_rank in b_ranks]) / n) ** 0.5
    return rank_covariance / (a_ranks_sd * b_ranks_sd + 1e-8)

def main(args, timer):
    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    args.world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'
    torch.autograd.set_detect_anomaly(True) 

    if args.device_id == 0:
        hostname = socket.gethostname()
        print("Hostname:", hostname)
        print(f"TrainConfig: {args}")
    timer.report("Setup for distributed training")

    data_path = f"/data/{args.dataset_id}"
    dataset = EVAL_HDF_Dataset(data_path)
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)
    timer.report(f"Intitialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test board evaluations.")

    train_sampler = InterruptableDistributedSampler(train_dataset)
    test_sampler = InterruptableDistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, sampler=test_sampler)
    timer.report("Prepared dataloaders")

    model_config = yaml.safe_load(open(args.model_config))
    if args.device_id == 0:
        print(f"ModelConfig: {model_config}")
    model_config["device"] = 'cuda'
    model = Model(**model_config)
    model = model.to(args.device_id)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    timer.report(f"Initialized model with {params:,} params, moved to device")

    model = DDP(model, device_ids=[args.device_id])
    timer.report("Prepared model for distributed training")

    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = {
        "train": MetricsTracker(), 
        "test": MetricsTracker(), 
        "best_rank_corr": float("-inf")
    }

    if args.is_master:
        writer = SummaryWriter(log_dir=os.environ["LOSSY_ARTIFACT_PATH"])

    output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    saver = AtomicDirectory(output_directory=output_directory, is_master=args.is_master)

    # set the checkpoint_path if there is one to resume from
    checkpoint_path = None
    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)
        checkpoint_path = os.path.join(latest_checkpoint_path, "checkpoint.pt")
    elif args.load_path:
        # assume user has provided a full path to a checkpoint to resume
        if os.path.isfile(args.load_path):
            checkpoint_path = args.load_path

    if checkpoint_path:
        timer.report(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer = checkpoint["timer"]
        timer.start_time = time.time()
        timer.report("Retrieved saved checkpoint")

    ## TRAINING
    
    for epoch in range(train_dataloader.sampler.epoch, 10_000):

        train_dataloader.sampler.set_epoch(epoch)
        test_dataloader.sampler.set_epoch(epoch)

        ## TRAIN
        
        timer.report(f"Training epoch {epoch}")
        train_batches_per_epoch = len(train_dataloader)
        train_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum)
        optimizer.zero_grad()
        model.train()

        for boards, scores in train_dataloader:

            # Determine the current step
            batch = train_dataloader.sampler.progress // train_dataloader.batch_size
            is_accum_batch = (batch + 1) % args.grad_accum == 0
            is_last_batch = (batch + 1) == train_batches_per_epoch
            is_save_batch = ((batch + 1) % args.save_steps == 0) or is_last_batch

            scores = logish_transform(scores) # suspect this might help
            boards, scores = boards.to(args.device_id), scores.to(args.device_id)

            logits = model(boards)
            loss = loss_fn(logits, scores) / args.grad_accum

            loss.backward()
            train_dataloader.sampler.advance(len(scores))

            # How accurately do our model scores rank the batch of moves? 
            rank_corr = spearmans_rho(logits, scores)

            metrics["train"].update({
                "examples_seen": len(scores),
                "accum_loss": loss.item() * args.grad_accum, # undo loss scale
                "rank_corr": rank_corr
            })

            if is_accum_batch or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
                step = batch // args.grad_accum
                
                # learning rate warmup
                lr_factor = min((epoch * train_steps_per_epoch + step) / args.ws, 1)
                next_lr = lr_factor * args.lr
                for g in optimizer.param_groups:
                    g['lr'] = next_lr
                
                metrics["train"].reduce()
                rpt = metrics["train"].local
                avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                rpt_rank_corr = 100 * rpt["rank_corr"] / ((batch % args.grad_accum + 1) * args.world_size)
                report = f"""\
Epoch [{epoch:,}] Step [{step:,} / {train_steps_per_epoch:,}] Batch [{batch:,} / {train_batches_per_epoch:,}] Lr: [{lr_factor * args.lr:,.3}], \
Avg Loss [{avg_loss:,.3f}], Rank Corr.: [{rpt_rank_corr:,.3f}%], Examples: {rpt['examples_seen']:,.0f}"""
                timer.report(report)
                metrics["train"].reset_local()

                if args.is_master:
                    total_progress = batch + epoch * train_batches_per_epoch
                    writer.add_scalar("train/learn_rate", next_lr, total_progress)
                    writer.add_scalar("train/loss", avg_loss, total_progress)
                    writer.add_scalar("train/batch_rank_corr", rpt_rank_corr, total_progress)

            # Saving
            if is_save_batch:
                checkpoint_directory = saver.prepare_checkpoint_directory()

                if args.is_master:
                    # Save checkpoint
                    atomic_torch_save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "metrics": metrics,
                            "timer": timer
                        },
                        os.path.join(checkpoint_directory, "checkpoint.pt"),
                    )
                
                saver.symlink_latest(checkpoint_directory)

        ## TESTING ##
        
        timer.report(f"Testing epoch {epoch}")
        test_batches_per_epoch = len(test_dataloader)
        model.eval()

        with torch.no_grad():
            for boards, scores in test_dataloader:

                # Determine the current step
                batch = test_dataloader.sampler.progress // test_dataloader.batch_size
                is_last_batch = (batch + 1) == test_batches_per_epoch
                is_save_batch = ((batch + 1) % args.save_steps == 0) or is_last_batch

                scores = logish_transform(scores) # suspect this might help
                boards, scores = boards.to(args.device_id), scores.to(args.device_id)

                logits = model(boards)
                loss = loss_fn(logits, scores)
                test_dataloader.sampler.advance(len(scores))

                # How accurately do our model scores rank the batch of moves? 
                rank_corr = spearmans_rho(logits, scores)

                metrics["test"].update({
                    "examples_seen": len(scores),
                    "accum_loss": loss.item(), 
                    "rank_corr": rank_corr
                })
                
                # Reporting
                if is_last_batch:
                    metrics["test"].reduce()
                    rpt = metrics["test"].local
                    avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                    rpt_rank_corr = 100 * rpt["rank_corr"] / (test_batches_per_epoch * args.world_size)
                    report = f"Epoch [{epoch}] Evaluation, Avg Loss [{avg_loss:,.3f}], Rank Corr. [{rpt_rank_corr:,.3f}%]"
                    timer.report(report)
                    metrics["test"].reset_local()

                    if args.is_master:
                        writer.add_scalar("test/loss", avg_loss, epoch)
                        writer.add_scalar("test/batch_rank_corr", rpt_rank_corr, epoch)
                
                # Saving
                if is_save_batch:
                    # force save checkpoint if test performance improves
                    if is_last_batch and (rpt_rank_corr > metrics["best_rank_corr"]):
                        force_save = True
                        metrics["best_rank_corr"] = rpt_rank_corr
                    else:
                        force_save = False

                    checkpoint_directory = saver.prepare_checkpoint_directory(force_save=force_save)

                    if args.is_master:
                        # Save checkpoint
                        atomic_torch_save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "train_sampler": train_dataloader.sampler.state_dict(),
                                "test_sampler": test_dataloader.sampler.state_dict(),
                                "metrics": metrics,
                                "timer": timer
                            },
                            os.path.join(checkpoint_directory, "checkpoint.pt"),
                        )
                    
                    saver.atomic_symlink(checkpoint_directory)

        train_dataloader.sampler.reset_progress()
        test_dataloader.sampler.reset_progress()
        

timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
