import os
import torch
import argparse

from lightning.pytorch import Trainer, seed_everything
from scripts.dataset import GLCliCCoAuthorDataset
from utils.config import GLCLiCModelConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scripts.model import GLCLiC


def build_config() -> GLCLiCModelConfig:
    return GLCLiCModelConfig(
        seed=42,
        dataset="CoAuthor",
        train_sources=None,
        test_sources=None,
        train_data_type="all",
        global_coherence=True,
        local_coherence=True,
        global_lexical=True,
        local_lexical=True,
        use_gpu=True if torch.cuda.is_available() else False,
        lr=1e-4,
        dropout=0.3,
    )


def compute_tokenizer_length_for_coauthor() -> int:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    # Match trainer logic: +5 special tokens for CoAuthor
    tokenizer_len = len(tokenizer) + 5
    return tokenizer_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="weights/detector_all/gl-clic.ckpt")
    parser.add_argument("--parsed_data_path", type=str, default="dataset/CoAuthor/parsed/coauthor_paraphrase.pkl")
    parser.add_argument("--preprocessed_data_path", type=str, default="dataset/CoAuthor/preprocessed/coauthor_paraphrase.pkl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Limit data for quick runs")
    args = parser.parse_args()

    seed_everything(42, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = build_config()
    tokenizer_len = compute_tokenizer_length_for_coauthor()

    # Load model from checkpoint
    model = GLCLiC.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        config=config,
        tokenizer_len=tokenizer_len,
        strict=True,
        map_location="cpu",
    )

    # Dataset and DataLoader for test set
    test_dataset = GLCliCCoAuthorDataset(
        parsed_data_file_name=args.parsed_data_path,
        preprocessed_data_file_name=args.preprocessed_data_path,
        stage="test",
        type="all",
        recreate=args.recreate,
        model_config=config,
        test=args.debug,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count() or 0,
    )

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False,
    )

    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
