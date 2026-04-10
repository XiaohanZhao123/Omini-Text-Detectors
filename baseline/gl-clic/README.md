# **GL-CLiC**: Global-Local Coherence and Lexical Complexity for Sentence-Level AI-Generated Text Detection
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11.12-green.svg)
![uv](https://img.shields.io/badge/uv-0.7.3-purple.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.4.0-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.46.3-yellow.svg)


## Abstract
Unlike document-level AI-generated text (AIGT) detection, sentence-level AIGT detection remains underexplored, despite its importance for addressing collaborative writing scenarios where humans modify AIGT suggestions on a sentence-by-sentence basis. Prior sentence-level detectors often neglect the valuable context surrounding the target sentence, which may contain crucial linguistic artifacts that indicate a potential change in authorship. We propose **GL-CLiC**, a novel technique that leverages both **G**lobal and **L**ocal signals of **C**oherence and **L**ex**i**cal **C**omplexity, which we operationalize through discourse analysis and CEFR-based vocabulary sophistication. **GL-CLiC** models local coherence and lexical complexity by examining a sentence's relationship with its neighbors or peers, complemented with its document-wide analysis. Our experimental results show that **GL-CLiC** achieves superior performance and better generalization across domains compared to existing methods.


## Framework Overview
![Framework](figures/framework.svg)
>The **GL-CLiC** framework overview that consists of six modules: Global Coherence (GC), Local Coherence (LC), Global Lexical complexity (GL), Local Lexical complexity (LL), Sentence Representations (SR), and Classifier (C).


## Project Directory Structure
```
├── dataset/              # Datasets folder
├── scripts/              # Contains scripts for model, preprocessing, trainer, etc.
├── utils/                # Helper scripts for parsing, features, etc.
├── env.example           # Environment variables example
├── train.py              # Main script for training GL-CLiC
├── pyproject.toml        # Project dependencies
└── README.md
```


## Dataset
We use two datasets in our experiments: CoAuthor and SeqXGPT. You can read the original papers here:
- [CoAuthor: Designing a Human-AI Collaborative Writing Dataset for Exploring Language Model Capabilities](https://dl.acm.org/doi/pdf/10.1145/3491102.3502030)
- [SeqXGPT: Sentence-Level AI-Generated Text Detection](https://aclanthology.org/2023.emnlp-main.73.pdf)


## Environment Variables
Please rename `env.example` to `.env` and fill the information.
```env
WANDB_API_KEY="Your wandb API Key"
```


## Dependencies
We use `uv` as our Python package and project manager. You can install `uv` by following the [official guide](https://docs.astral.sh/uv/getting-started/installation/). To run our experiments please follow these steps after you install `uv`.
\
**Environment setup**:
```shell
uv venv --python $(cat .python-version)
```
**Installing dependencies**:
```shell
uv sync
```


## Usage
### GL-CLiC
To run the experiment, use the following command:
```shell
uv run train.py
```

#### Parameters
- `dataset`: dataset used for experiment, you can choose between ["CoAuthor", "SeqXGPT-Bench"]
- `raw_data_path`: path to the raw dataset
- `parsed_data_path`: path to the parsed dataset
- `preprocessed_data_path`: path to the preprocessed dataset
- `recreate`: to re-parse and re-preprocess the dataset (by default it will use the saved parsed & preprocessed dataset)
- `type`: data domain type for "CoAuthor" dataset, you can choose between ["creative", "argumentative", "all"]
- `train_sources`: train data source for "SeqXGPT-Bench" dataset, you can choose multiple from ["gpt2", "gpt3", "gptj", "gptneo", "human", "llama"]
- `test_sources`: test data source for "SeqXGPT-Bench" dataset, you can choose multiple from ["gpt2", "gpt3", "gptj", "gptneo", "human", "llama"]
- `batch_size`: batch size for training (default is 2)
- `max_epochs`: max epochs for training (default is 10)
- `learning_rate`: learning rate for training (default is 1e-4)
- `dropout`: dropout for training (default is 0.3)
- `alpha`: alpha for the loss (default is 1.0)
- `model_save_path`: path for saving model weight
- `log_path`: path for saving training log
- `global_coherence` or `gc`: to activate global coherence feature
- `local_coherence` or `lc`: to activate local coherence feature
- `global_lexical` or `gl`: to activate global lexical feature
- `local_lexical` or `ll`: to activate local lexical feature

#### Example
```shell
uv run train.py --dataset CoAuthor -gc -lc -gl -ll
```

## Citation
If you find our codes or paper useful, please consider giving us a star or citing our work.

```bibtex
@inproceedings{adi-etal-2025-gl,
    title = "{GL}-{CL}i{C}: Global-Local Coherence and Lexical Complexity for Sentence-Level {AI}-Generated Text Detection",
    author = "Adi, Rizky  and
      Irnawan, Bassamtiano Renaufalgi  and
      Suzuki, Yoshimi  and
      Fukumoto, Fumiyo",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.188/",
    pages = "3600--3617",
    ISBN = "979-8-89176-298-5"
}
```
