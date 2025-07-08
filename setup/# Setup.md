# Setup

# Running FSDL Text Recognizer 2021 Labs in Google Colab
Deep learning requires access to accelerated computation hardware, specifically NVIDIA GPUs or Google TPUs.

If you have access to a computer with an NVIDIA GPU running Linux, you can set it up for local use.  
If not, you can compute using **Google Colab**.

If you'd like to use Google Cloud Platform, fullstack deep learning from [Berkely](fullstackdeeplearning.com) have a tutorial for setting up an AI Platform Jupyter Lab instance for computing, which works quite well. A well illustrated [tutorial](https://docs.google.com/document/d/1mSB_p1Chxg6IGYbuRxgPSA3Ps6BjhBZV7Ti3W_Qx0Ws/) for setting up an AI Platform Jupyter Lab instance for computing, which works quite well.


---

## Colab

**[Watch As Video](https://www.loom.com/share/9c99c49fb9ca456bb0e56ccc098ae87a)**

Google Colab is a great way to get access to fast GPUs for free. All you need is a Google account (Berkeley accounts will work too).

1. Go to [Google Colab](https://colab.research.google.com) and create a **New Notebook**.

2. Connect your notebook to a GPU runtime:  
   **Runtime > Change Runtime type > GPU**.

   !Connect your new notebook to a GPU runtime by doing Runtime > Change Runtime type > GPU.
![](colab_runtime.png)

3. Run the following in the first cell to check your GPU:

   ```python
   !nvidia-smi
   ```

   You should see a table showing your GPU. :)

4. Paste the following into a cell and run it:
this is the full stack deeplearning lab 1 materials and its associate repository

   ```python
   # FSDL Spring 2021 Setup
   !git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs
   %cd fsdl-text-recognizer-2021-labs
   !pip3 install boltons wandb pytorch_lightning==1.1.4
   !pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
   %env PYTHONPATH=.:$PYTHONPATH
   ```

  This will check out the lab repository, `cd` into it, install the only missing Python package (`pytorch_lightning` -- everything else, like PyTorch itself, comes pre-installed on Colab), and allows Python to find packages in the current working directory.

Now we can enter the `lab1` directory and make sure things work:
![](colab_lab1.png)
---

## Colab Pro

You may be interested in signing up for [Colab Pro](https://colab.research.google.com/signup).

For $10/month, you get:
- Priority access to fast GPUs (e.g. V100 vs K80) and TPUs
- 24h rather than 12h runtime
- More RAM

---

## VSCode on Google Colab (Advanced)

It is possible to use the VSCode interface in Colab.

1. Open a Colab notebook, connect to your desired runtime, and run:

   ```python
   # Launch VSCode server
   !curl -fsSL https://code-server.dev/install.sh | sh
   !nohup code-server --port 9000 --auth none &
   
   # Tunnel its IP using ngrok
   !pip install pyngrok
   from pyngrok import ngrok
   # ngrok.set_auth_token("get from https://dashboard.ngrok.com/auth/your-authtoken, if you want to pay $10/month for a little bit better service")
   url = ngrok.connect(9000)
   print(url)
   ```

   You should see something like this:

   ![](colab_vscode.png)

2. Clicking the ngrok link takes you to a web VSCode interface:

   ![VSCode in browser via ngrok]:
   ![](colab_vscode_2.png)

You can sign up for a paid version of ngrok ($10/month) for HTTPS tunneling and a slightly nicer experience.

---

## Local Setup

Setting up a machine you can sit in front of or SSH into is easy.

### 1. Check out the repo

```bash
git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs.git
cd fsdl-text-recognizer-2021-labs
```

### 2. Set up the Python environment

We use **conda** for managing Python and CUDA versions, and **pip-tools** for managing Python package dependencies.

First, install the Python + CUDA environment using Conda:

- [Install conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)

After installing, restart your terminal and ensure you can run the `conda` command.

Run:

```bash
make conda-update
```

This creates an environment called `fsdl-text-recognizer-2021` as defined in `environment.yml`.

Next, activate the conda environment:

```bash
conda activate fsdl-text-recognizer-2021
```

> **IMPORTANT:** Every time you work in this directory, start your session with `conda activate fsdl-text-recognizer-2021`.

---

### 3. Install Python packages

Install all necessary Python packages:

```bash
make pip-tools
```

Using pip-tools lets us:
- Separate dev from production dependencies (`requirements-dev.in` vs `requirements.in`)
- Have a lockfile of exact versions for all dependencies (`requirements-dev.txt` and `requirements.txt`)
- Easily deploy to targets that may not support the conda environment

If you add, remove, or need to update versions of some requirements, edit the `.in` files and run `make pip-tools` again.

---

### 4. Set PYTHONPATH

Run:

```bash
export PYTHONPATH=.:$PYTHONPATH
```

To avoid setting this every time, add it as the last line of your `~/.bashrc` file:

```bash
echo 'export PYTHONPATH=.:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```
---

## DEBUGGING

its possible to run into errors when running the installation on google colab when you run the fsdl command above due to compablities issues of pytorch and cuda version. this is due to the fact that this lab was pre-trained and uses old versions of cuda and pytorch which are now outdaded. Due to the age of the original setup instructions and the evolving nature of deep learning libraries, running these labs directly on modern Google Colab instances can lead to dependency conflicts, especially with PyTorch and FastAI.

This debugging guide below provides a revised installation process that has been tested and confirmed to work as of [Current Date: July 8, 2025].

---

**Problem:**
The original lab setup (e.g., `pip install torch==1.7.1+cu110 ...`) attempts to install outdated PyTorch versions that are no longer available or compatible with current Google Colab CUDA environments. This results in errors like `ERROR: No matching distribution found for torch==1.7.1+cu110`.

**Solution:**
We will manually install compatible versions of PyTorch and FastAI, leveraging Colab's current CUDA setup.

**Step-by-Step Installation in Google Colab:**

1.  **Open a new Google Colab notebook.**

2.  **Verify Colab's CUDA Version:**
    First, let's confirm the CUDA version available on your Colab runtime.
    ```python
    # Run this in a Colab code cell
    !nvcc --version
    ```
    *Expected output will indicate the CUDA version, e.g., `release 12.5`.* This is important for selecting the correct PyTorch build.

3.  **Clean Up Existing (Potentially Conflicting) Installations:**
    It's crucial to remove any pre-installed or previously attempted installations of these libraries to ensure a clean slate.
    ```python
    # Run this in a Colab code cell
    !pip uninstall torch torchvision torchaudio fastai -y
    !pip uninstall pytorch_lightning -y # Optional: Uninstall old pytorch_lightning if present
    ```

4.  **Install Compatible PyTorch and FastAI Versions:**
    Based on Colab's typical CUDA versions (e.g., 12.5) and FastAI's requirements (`fastai==2.7.19` needs `torch<2.7`), we've found that `torch==2.6.0` built for CUDA 12.6 is generally compatible.
    ```python
    # Run this in a Colab code cell
    # Install PyTorch 2.6.0 with CUDA 12.6 support
    !pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

    # Reinstall FastAI 2.7.19, which is compatible with PyTorch 2.6.0
    !pip install fastai==2.7.19

    # Install other necessary lab dependencies
    !pip install boltons wandb
    # Note: The original lab might specify pytorch_lightning==1.1.4.
    # This version is very old and often incompatible with newer PyTorch.
    # If a specific module from pytorch_lightning is later missing or causes errors,
    # consider installing a slightly newer compatible version like pytorch_lightning==1.8.6
    # if necessary for specific lab functionalities.
    ```

5.  **Restart the Colab Runtime:**
    This step is essential for the newly installed packages to be properly loaded into your environment.
    * Go to `Runtime` -> `Restart runtime` from the top menu in Google Colab.

6.  **Clone the Repository and Set Python Path:**
    After the runtime restarts, execute these commands in a *new* Colab code cell.
    ```python
    # Run this in a Colab code cell (after runtime restart)
    !git clone [https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs)
    %cd fsdl-text-recognizer-2021-labs
    %env PYTHONPATH=.:$PYTHONPATH
    ```

7.  **Verify Your Installation:**
    Confirm that PyTorch and FastAI are correctly installed and that GPU acceleration is enabled.
    ```python
    # Run this in a Colab code cell
    import torch
    import fastai

    print(f"PyTorch version: {torch.__version__}")
    print(f"FastAI version: {fastai.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    ```
    *Expected Output:*
    * `PyTorch version: 2.6.0+cu126` (or similar for 2.6.x)
    * `FastAI version: 2.7.19`
    * `CUDA available: True` (This confirms GPU is recognized and working with PyTorch)

You should now be able to run the lab notebooks successfully!

---

## Summary

- `environment.yml` specifies python and optionally cuda/cudnn
- `make conda-update` creates/updates the conda env
- `conda activate fsdl-text-recognizer-2021` activates the conda env
- `requirements/prod.in` and `requirements/dev.in` specify python package requirements
- `make pip-tools` resolves and installs all Python packages
- Add `export PYTHONPATH=.:$PYTHONPATH` to your `~/.bashrc` and source it

---