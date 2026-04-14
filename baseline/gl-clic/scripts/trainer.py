import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import wandb
import datetime


from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from scripts.data_module import GLCliCCoAuthorDataModule, GLCliCSeqXGPTBenchDataModule
from lightning.pytorch import Trainer, seed_everything
from utils.config import GLCLiCModelConfig
from transformers import AutoTokenizer
from dataclasses import dataclass
from scripts.model import GLCLiC


seed_everything(42, workers=True)


@dataclass
class GLCLiCTrainerConfig:
    test: bool
    dataset: str
    train_sources: list
    test_sources: list
    wandb_project: str
    raw_data_path: str
    parsed_data_path: str
    preprocessed_data_path: str
    recreate: bool
    experiment_type: str
    batch_size: int
    max_epochs: int
    learning_rate: float
    dropout: float
    alpha: float
    model_save_path: str
    log_path: str
    global_coherence: bool
    local_coherence: bool
    global_lexical: bool
    local_lexical: bool

valid_seqxgpt_bench_sources = [
    "gpt2",
    "gpt3",
    "gptj",
    "gptneo",
    "human",
    "llama",
]


class GLCLiCTrainer:
    def __init__(self, config: GLCLiCTrainerConfig):

        print("\n[TRAINER] Initialization...\n")
        global_coherence_feature = config.global_coherence if config.global_coherence else False
        local_coherence_feature = config.local_coherence if config.local_coherence else False
        global_lexical_feature = config.global_lexical if config.global_lexical else False
        local_lexical_feature = config.local_lexical if config.local_lexical else False

        if not global_coherence_feature and not local_coherence_feature and not global_lexical_feature and not local_lexical_feature:
            raise ValueError("At least one of global_coherence_feature, local_coherence_feature, global_lexical_feature, or local_lexical_feature must be set to True")

        if config.dataset not in ["CoAuthor", "SeqXGPT-Bench"]:
            raise ValueError("Invalid dataset, please choose one of 'SimLLM', 'CoAuthor', or 'SeqXGPT-Bench'")

        if config.dataset == "SeqXGPT-Bench":
            if config.train_sources is None or config.test_sources is None:
                raise ValueError("Please provide train and test sources")
            for source in config.train_sources + config.test_sources:
                if source not in valid_seqxgpt_bench_sources:
                    raise ValueError(f"Invalid source: {source}. Please choose from {valid_seqxgpt_bench_sources}")

        print(f"Train sources: {config.train_sources}")
        print(f"Test sources: {config.test_sources}")

        print("\n[TRAINER] Setting up directories...\n")
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.log_path), exist_ok=True)

        if config.test:
            print("\n[TRAINER] Running in test mode...\n")
        elif os.getenv('WANDB_API_KEY') is None:
            print("\n[TRAINER] WANDB_API_KEY not found in environment variables, skipping wandb logging...\n")
        else:
            print("\n[TRAINER] Initializing wandb...\n")
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S")
            wandb_run_name = f"{config.experiment_type.title()} | "
            if global_coherence_feature:
                wandb_run_name += "GC-"
            if local_coherence_feature:
                wandb_run_name += "LC-"
            if global_lexical_feature:
                wandb_run_name += "GL-"
            if local_lexical_feature:
                wandb_run_name += "LL-"
            wandb_run_name = wandb_run_name[:-1]
            wandb_run_name = f"{wandb_run_name} | {now}"
            self.wandb_logger = WandbLogger(project=config.wandb_project, name=wandb_run_name, log_model=False, save_dir=f"{config.log_path}_{config.experiment_type}")

        print("\n[TRAINER] Initializing Model Config...\n")
        detector_model_config = GLCLiCModelConfig(
            seed=42,
            dataset=config.dataset,
            train_sources=config.train_sources,
            test_sources=config.test_sources,
            train_data_type=config.experiment_type,
            global_coherence=global_coherence_feature,
            local_coherence=local_coherence_feature,
            global_lexical=global_lexical_feature,
            local_lexical=local_lexical_feature,
            use_gpu=True if torch.cuda.is_available() else False,
            lr=config.learning_rate,
            dropout=config.dropout,
            alpha=config.alpha
        )

        print("\n[TRAINER] Initializing data module...\n")
        if config.dataset == "CoAuthor":
            self.data_module = GLCliCCoAuthorDataModule(
                model_config=detector_model_config,
                raw_data_path=config.raw_data_path,
                parsed_data_path=config.parsed_data_path,
                preprocessed_data_path=config.preprocessed_data_path,
                essay_type=config.experiment_type,
                batch_size=config.batch_size,
                recreate=config.recreate,
                test=config.test
            )

        if config.dataset == "SeqXGPT-Bench":
            self.data_module = GLCliCSeqXGPTBenchDataModule(
                model_config=detector_model_config,
                raw_data_path=config.raw_data_path,
                parsed_data_path=config.parsed_data_path,
                preprocessed_data_path=config.preprocessed_data_path,
                batch_size=config.batch_size,
                recreate=config.recreate,
                test=config.test
            )



        print("\n[TRAINER] Initializing Model...\n")
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        tokenizer_len = len(tokenizer) + 5 # 5 for special tokens
        detector_model_config.print_config()
        self.detector_model = GLCLiC(detector_model_config, tokenizer_len)
        del tokenizer

        print("\n[TRAINER] Initializing Trainer...\n")
        early_stopping = EarlyStopping(monitor='val_macro_f1_score', patience=3, mode='max', min_delta=0.005)
        checkpoint = ModelCheckpoint(dirpath=f"{config.model_save_path}_{config.experiment_type}", monitor='val_macro_f1_score', mode='max', save_top_k=1)
        csv_logger = CSVLogger(f"{config.log_path}_{config.experiment_type}")

        if config.test:
            logger = [csv_logger]
        elif os.getenv('WANDB_API_KEY') is None:
            logger = [csv_logger]
        else:
            if config.dataset == "CoAuthor":
                self.wandb_logger.experiment.config.update({
                    "dataset": config.dataset,
                    "batch_size": config.batch_size,
                    "max_epochs": config.max_epochs,
                    "lr": config.learning_rate,
                    "dropout": config.dropout,
                    "alpha": config.alpha,
                    "global_coherence": global_coherence_feature,
                    "local_coherence": local_coherence_feature,
                    "global_lexical": global_lexical_feature,
                    "local_lexical": local_lexical_feature,
                    "type": config.experiment_type,
                })
            if config.dataset == "SeqXGPT-Bench":
                self.wandb_logger.experiment.config.update({
                    "dataset": config.dataset,
                    "batch_size": config.batch_size,
                    "max_epochs": config.max_epochs,
                    "lr": config.learning_rate,
                    "dropout": config.dropout,
                    "alpha": config.alpha,
                    "global_coherence": global_coherence_feature,
                    "local_coherence": local_coherence_feature,
                    "global_lexical": global_lexical_feature,
                    "local_lexical": local_lexical_feature,
                    "train_sources": config.train_sources,
                    "test_sources": config.test_sources,
                })
            logger = [self.wandb_logger, csv_logger]

        self.trainer = Trainer(
            deterministic=True,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            max_epochs= 1 if config.test else config.max_epochs,
            callbacks=[early_stopping] if config.test else [early_stopping, checkpoint],
            logger=logger,
        )

        self.config = config


    def train(self):
        print("\n[TRAINER] Begin Training...\n")
        self.trainer.fit(self.detector_model, self.data_module)


    def test(self):
        print("\n[TRAINER] Begin Testing...\n")
        self.trainer.test(self.detector_model, self.data_module, ckpt_path="best")


    def finish(self):
        if not self.config.test:
            print("\n[TRAINER] Finalizing...\n")
            self.wandb_logger.finalize(status="success")
            self.wandb_logger.experiment.finish()
