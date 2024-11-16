# chess-hackathon-4

## Quick Start Guide
Before attempting the steps in this guide, please ensure you have completed all onboarding steps from the **Getting Started** section of the [Strong Compute Developer Docs](https://strong-compute.gitbook.io/developer-docs). 

### Step 1. Installation
Create and source a new python virtual environment.

```
python3 -m virtualenv ~/.chess
source ~/.chess/bin/activate
```

Clone this repo and install the requirements.

```
cd ~
git clone https://github.com/StrongResearch/chess-hackathon-4.git
cd ~/chess-hackathon-4
pip install -r requirements.txt
```

### Step 2. Choose a model
1. Nagivate to the **models** subdirectory of this repository.
2. Decide whether you want to train a **chessGPT** or **chessVision** model.
3. Navigate to the appropriate model type subdirectory for your chosen model type.
4. The model type subdirectory will contain two further subdirectories, one for each example model of this type. Decide which of the two example models you want to train.

### Step 3. Copy necessary training files to repository root
Copy the following files from the **model type** subdirectory to the root directory for this repo (i.e. copy from `chess-hackathon-4/models/chessVision` to `chess-hackathon-4`).
 - `<type>.isc`
 - `train_<type>.py`

 Copy the following files from the **example model** subdirectory to the root directory for this repo (i.e. copy from `chess-hackathon-4/models/chessVision/conv` to `chess-hackathon-4`)
 - `model.py`
 - `model_config.yaml`

### Step 4. Update the experiment launch file
Update the experiment launch file (the `chessGPT.isc` or `chessVision.isc`) you chose, with your Project ID.

The `<type>.isc` file is prepared with a suitable dataset already, but if you want to select another dataset (see below) you can also update the Dataset ID.

### Step 5. Launch your experiment
Launch your experiment with the following.

```
isc train <type>.isc
```

### Step 6. Validate your model inference
- In your terminal, run `isc experiments` to obtain the output path for the experiment you launched.
- Wait for your experiment to reach the status `completed` (re-run `isc experiments` until you see `completed`).
- Navigate to the output path for your experiment and copy the `checkpoint.pt` from within the `/latest_pt` subdirectory into the home directory for this repo (i.e. `/root/chess-hackathon-4`).
- In your terminal, navigate to the home directory for this repo with `cd /root/chess-hackathon-4` and run `python pre_submission_val.py`.
This will validate that your model is able to initialize correctly, load the checkpoint, and infer fast enough to play in the tournament, and is an important step **before launching burst**. Otherwise, you might develop a model and spend time training it only to discover that it is too big, and you will need to train a smaller model instead. 

For more information about this see below under **Pre-submission model validation**.

### Step 7. Launch your experiment to train with `compute_mode = "burst"`
Once your model has successfully `completed` a run with `compute_mode = "cycle"` you will have confidence that it will run successfully on a dedicated cluster. Your next step is update your experiment launch file with `compute_mode = "burst"` and again run `isc train <type>.isc`.

This time you will see a message directing you to Control Plane to launch your burst experiment. Visit the Experiments page on Control Plane and click "Launch Burst" next to your experiment.

Click on the "View" button for your experiment in Control Plane to follow progress initializing your experiment to run on a dedicated cluster. Be patient, this can take a few minutes.

Once your experiment reaches the state of `running`, visit the User Credentials page in Control Plane and click **Stop** on your container, then click **Start** on your container again. When your container is started again, you will find artefacts from your experiment training on its dedicated cluster sycning to a directory in `/root/exports/<experiment-id>/outputs`. Interacting with this directory is slow because it is a mounted bucket - again please be patient. To track performance metrics logging to `rank_0.txt` or access checkpoints, copy the files you need from `/root/exports/<experiment-id>/outputs` to another subdirectory in `/root` beforehand.

### Step 8. Resume training your model from a previous checkpoint
If your experiment stops with status `strong_fail`, or if you **Stop** your experiment via the CLI or Control Plane, then you may be able to **resume** training your experiment from its most recent checkpoint.

The training scripts included in this repo under `/chess-hackathon-4/models` implement an optional argument `--load-path`. Include this argument in your experiment launch file as follows, passing in the path to the most recent checkpoint from the stoppped experiment.

```toml
isc_project_id = "<project-id>"
experiment_name = "vision"
gpu_type = "24GB VRAM GPU"
gpus = 48
output_path = "~/outputs/vision"
dataset_id = "96f6d30d-3dec-474b-880e-d2fa3ba3756e"
compute_mode = "cycle"
command = '''
source ~/.chess/bin/activate &&
cd ~/chess-hackathon-4/ &&
torchrun --nnodes=$NNODES --nproc-per-node=$N_PROC --master_addr=$MASTER_ADDR
--master_port=$MASTER_PORT --node_rank=$RANK
train_chessVision.py --load-path /root/<path>/<to>/checkpoint.pt'''
```

You can then launch a new experiment with `isc train <type>.isc` which will resume training from that checkpoint.

**Note: when resuming from `comput_mode = "burst"` experiments, ensure you have copied the most recent checkpoint out of the `/root/exports` directory into another location in `/root` before resuming your experiment.**

## Inference (game play)
To understand how your model will be instantiated and called during gameplay, refer to the `gameplay.ipynb` notebook.

## Important Rules & Submission Spec
### Important rules
You may develop most any kind of model you like, but your submission must adhere to the following rules. 
 - Your submission must conform to the specification (below),
 - Your model must pass the pre-submission validation check (below) to be admitted into the tournament, 
 - Your model must be trained **entirely from scratch** using the provided compute resources. 
 - You **may not** use pretrained models (this includes no transfer learning, fine-tuning, or adaptation modules).
 - You **may not** hard-code any moves (e.g. no opening books).
 - Your model **must** use or be compatible with the dependencies included in the `requirements.txt` file for this repo. You may install other additional dependencies for the purpose of **training** but for inference (e.g. game play / tournament) your model **must not** require any dependencies other than those included in the `requirements.txt` file.

### Submission specification
Your submission must follow the following directory structure. Ensure you have moved your `model.py`, `model_config.yaml`, and `checkpoint.pt` files into a **separate sub/directory**. Then copy in `pre_submission_val.py` and `chess_gameplay.py` and run this script with `python pre_submission_val.py` to test that your model will build and infer within the allowed time. For more infro
```
└─team-name
    ├─ model.py
    ├─ model_config.yaml
    ├─ checkpoint.pt
    ├─ pre_submission_val.py
    └─ chess_gameplay.py
```
**Do not make any changes to the contents of `pre_submission_val.py` or `chess_gameplay.py`**.

#### Specification for model_config.yaml
 - The `model_config.yaml` file must conform to standard yaml syntax.
 - The `model_config.yaml` file must contain all necessary arguments for instantiating your model. See below for demonstration of how the `model_config.yaml` is expected to be used during the tournament.

#### Specification for model.py
 - The `model.py` file must contain a class description of your model, which must be a PyTorch module called `Model`.
 - The `Model` class **must be self-contained**. All code necessary to instantiate your model should be included in the `model.py` file and dependencies installed with `requirements.txt`. Your `model.py` file **must not** import from any ancillary files in your project directory.
 - The model must not move any weights to the GPU upon initialization, it will be expected to run **entirely on the CPU** during the tournament.
 - The model must implement a `score` method. 
 - The `score` method must accept as input the following two positional arguments:
  1. A PGN string representing the current game up to the most recent move, and
  2. A string representing a potential next move.
 - The `score` method must return a `float` value which represents a score for the potential move given the PGN, where higher positive scores always indicate preference for selecting that move.
 - The model **must not** require GPU access to execute the `score` method.

#### Specification for checkpoint.pt
 - The `checkpoint.pt` file must be able to be loaded with the `torch.load` function.
 - Your model state dictionary must be able to be obtained from the loaded checkpoint object by calling `checkpoint[“model”]`.

#### Pre-submission model validation
Your model must satisfy the pre-submission validation check to gain admittance into the tournament. You can run the pre-submission validation check 
with the following.

```
python pre_submission_val.py
```

If successful, this test will return the following.

```
Outputs pass validation tests.
Model passes validation test.
```

If any errors are reported, your model has **failed the test** and must be amended in order to be accepted into the tournament.

## Know your datasets
There are four datasets that have been published for this hackathon which can be found on the **Datasets** page of **Control Plane** under **Public Datasets**.
1. `Hackathon 3 - PGN - Grand Master Games` (ID: `b90f0e85-2cd9-4909-8fce-af10dbaa95d7`)
2. `Hackathon 3 - PGN - Leela Chess Zero Training Test 60` (ID: `9a921d78-e7bc-4cf4-9e4a-7a3bfe890852`)
3. `Hackathon 3 - EVAL - Grand Master Games` (ID: `7d959dc4-f5f1-4aae-8e62-c53ece32876f`)
4. `Hackathon 3 - EVAL - Leela Chess Zero Training Test 60` (ID: `d1851b32-7b47-4e25-8c96-b39bb759d3d0`)

The `PGN` datasets are suitable for `chessGPT` model training. The `EVAL` datasets are suitable for `chessVision` model training. Choose a dataset that is suitable for your chosen model and note the Dataset ID.

A further two datasets have also been prepared which contain both of the above `PGN` and both of the above `EVAL` datasets respectively. Those datasets are named as follows.
1. `Hackathon 3 - PGN - Combined`
2. `Hackathon 3 - Eval - Combined`

Please note the training scripts published in this repo will not work with these two combined datasets without adjustment. You will need to update the training scripts, or write your own, to work with the above combined datasets if you wish.

All code used to develop these datasets can be found in `chess-hackathon-4/utils/data_preprocessing`. The `Hackathon 3 - PGN - Grand Master Games` dataset was generated using `gm_preproc.ipynb` notebook. The `Hackathon 3 - PGN - Leela Chess Zero Training Test 60` dataset was generated using `lc0_preproc.ipynb` notebook. The `Hackathon 3 - EVAL - Grand Master Games` and `Hackathon 3 - EVAL - Leela Chess Zero Training Test 60` datasets were generated by running a distributed processing workload with `preproc_boardeval.py` launched with `preproc.isc`, and post-processed with `eval_preproc.ipynb`.
