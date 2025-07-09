# miniLLM

### Table of contents
* [Introduction](#star2-introduction)
* [Installation](#wrench-installation)
* [How to run](#zap-how-to-run) 
* [Contact](#raising_hand-questions)

## :star2: Introduction

* <p align="justify">Designed and implemented a lightweight Transformer-based language model from scratch using PyTorch.</p>
* <p align="justify">Modular architecture with embedding layer, positional encoding, self-attention, and transformer blocks.</p>
* <p align="justify">Implemented tokenization and training pipeline for next-word prediction.</p>

![demo104](/images/demo.PNG)

Figure: *input "The first thing he wanted to do was get the lower" and num_predictions 7*

## :wrench: Installation

<p align="justify">Step-by-step instructions to get you running miniLLM:</p>

### 1) Clone this repository to your local machine:

```bash
git clone https://github.com/hoangnguyen2003/miniLLM.git
```

A folder called `miniLLM` should appear.

### 2) Install the required packages:

Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://www.anaconda.com/docs/getting-started/miniconda/install).

<p align="justify">You can re-create our conda enviroment from `environment.yml` file:</p>

```bash
cd miniLLM
conda env create --file environment.yml
```

<p align="justify">Your conda should start downloading and extracting packages.</p>

### 3) Activate the environment:

Your environment should be called `miniLLM`, and you can activate it now to run the scripts:

```bash
conda activate miniLLM
```

## :zap: How to run 
<p align="justify">To train miniLLM:</p>

```bash
python main.py --mode train
```

You can predict the next word directly from the command line:

```bash
python main.py --mode predict --input "The first thing he wanted to do was get the lower" --num_predictions 7
```

## :raising_hand: Questions
If you have any questions about the code, please contact Hoang Van Nguyen (hoangvnguyen2003@gmail.com) or open an issue.