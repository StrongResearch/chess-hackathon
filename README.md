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
Update the experiment launch files with your Project ID. The `<type>.isc` file is prepared with a suitable dataset already, but if you want to select another dataset (see below) you can also update the Dataset ID.

### Step 5. Launch your experiment
Launch your experiment with the following.

```
isc train <type>.isc
```

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
Your submission must follow the following directory structure. Ensure you have moved your `model.py`, `model_config.yaml`, and `checkpoint.pt` files into a **separate sub/directory**. Then copy in `pre_submission_val.py` and `chess_gameplay.py` and run this script with `python pre_submission_val.py` to test that your model will build and infer within the allowed time. **Do not make any changes to the contents of `pre_submission_val.py` or `chess_gameplay.py`**.

```
└─team-name
    ├─ model.py
    ├─ model_config.yaml
    ├─ checkpoint.pt
    ├─ pre_submission_val.py
    └─ chess_gameplay.py
```

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
python pre_submission_validation.py
```

If successful, this test will return the following.

```
Outputs pass validation tests.
Model passes validation test.
```

If any errors are reported, your model has **failed the test** and must be amended in order to be accepted into the tournament.

## Know your datasets
There are four datasets that have been published for this hackathon. 
1. `Hackathon 3 - PGN - Grand Master Games` (ID: `b90f0e85-2cd9-4909-8fce-af10dbaa95d7`)
2. `Hackathon 3 - PGN - Leela Chess Zero Training Test 60` (ID: `9a921d78-e7bc-4cf4-9e4a-7a3bfe890852`)
3. `Hackathon 3 - EVAL - Grand Master Games` (ID: `7d959dc4-f5f1-4aae-8e62-c53ece32876f`)
4. `Hackathon 3 - EVAL - Leela Chess Zero Training Test 60` (ID: `d1851b32-7b47-4e25-8c96-b39bb759d3d0`)

The `PGN` datasets are suitable for `chessGPT` model training. The `EVAL` datasets are suitable for `chessVision` model training. Choose a dataset that is suitable for your chosen model and note the Dataset ID.

All code used to develop these datasets can be found in `chess-hackathon-4/utils/data_preprocessing`. The `Hackathon 3 - PGN - Grand Master Games` dataset was generated using `gm_preproc.ipynb` notebook. The `Hackathon 3 - PGN - Leela Chess Zero Training Test 60` dataset was generated using `lc0_preproc.ipynb` notebook. The `Hackathon 3 - EVAL - Grand Master Games` and `Hackathon 3 - EVAL - Leela Chess Zero Training Test 60` datasets were generated by running a distributed processing workload with `preproc_boardeval.py` launched with `preproc.isc`, and post-processed with `eval_preproc.ipynb`.
