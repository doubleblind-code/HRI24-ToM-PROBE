# TOM-PROBE

*< NOTE > This repository is currently hosted anonymously adhering to double-blind conference reviewing process. On acceptance, we will host this under our institution's name.*

"TOM-PROBE: Theory of Mind - Perceived RObot Behavior rEcognition" is a dateset that can be useful for evaluating an LLM's (Large Language Model's) ToM capabilities based on its understanding of the behavior of an AI agent and how the shown behavior would be perceived by the human observer in the loop. Specifically we focus on four behavior types, namely - explicable, legible, predictable and obfuscatory behavior, which have been extensively used to synthesize interpretable robot behaviors.

*The dataset in this repository consists of .txt files, which can be used for generating responses from any LLM. Currently, we show the usage by using GPT-4 as a case study.*

## Installation

TOM-PROBE runs on Python 3.9.1. To install the required packages, run the following command:

```cmd
conda create --name <env> --file requirements.txt
git clone <repo>
```

## Dataset Strcture

- `data/` contains all the text files for querying LLMs.
- `queries/` contains the queries for each of the four behavior types (generated for prompting GPT-4).
- `results/` contains the results of the queries (generated by GPT-4).
- `conversation.py` contains the code for querying GPT-4 (or any OpenAI model.)
- `runner.py` contains the code for running the prompts for the four behavior types.
- `runner_conviction.py` contains the code for running the prompts for the conviction test.
- `runner_evaluation.py` contains the code for evaluating the generated responses.
- `runner_evaluation_conviction.py` contains the code for evaluating the generated responses for the conviction test.

Overall structure can be seen below:

```cmd
──data
    ├── envs_updated
        ├── vanilla
           ├── binary
                ├── exp
                ├── leg
                ├── obs
                └── pre
            |── reason
                ├── exp
                ├── leg
                ├── obs
                └── pre
        ├── uninformative_vanilla
            ├── ...
        ├── inconsistent
            ├── ...
── queries
    ├── ...

── results
    ├── ...

── conversation.py
── runner.py
── runner_conviction.py
── runner_evaluation.py
── runner_evaluation_conviction.py
```

## Supported Tests

1. Standard Probe Test (`vanilla`)
2. Robustness Tests:

    - Uninformative Probe Test (`uninformative_vanilla`)
    - Inconsistent Probe Test (`inconsistent`)
    - Conviction Probe Test  (`python runner_conviction.py`) # uses the same parameters as `runner.py`


## Example Usage

Please refer to `conversation.py` for adding your own OpenAI API key.

### Running the prompts for binary (Yes/No) and MCQ Reasoning questions:

```cmd
python runner.py
```

In `runner.py`, you can specify the following parameters:

```python
expt_name = "vanilla"  # Change this to "uninformative_vanilla" or "inconsistent".
models = ["gpt4"] # Change this to any other OpenAI LLM.
TEMPERATURE = 0 # GPT-4 supports temperature in the range [0, 2].
classification_dir = f'./data/envs_updated/{expt_name}/binary/'
reasoning_dir      = f'./data/envs_updated/{expt_name}/reason/'  
```

Note, for evaluating `inconsistent` test prompts, change the `classification_solutions`. See `runner.py` for more details. This also applies to `runner_evaluation.py` which generates more comprehensive evaluation metrics.

### For evaluation of the generated responses:

```cmd
python runner_evaluation.py
```

