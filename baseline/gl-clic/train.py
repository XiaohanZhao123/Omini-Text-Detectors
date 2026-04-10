import os
import nltk
import argparse
import torch

from lightning.pytorch import seed_everything
from scripts.trainer import GLCLiCTrainerConfig, GLCLiCTrainer


nltk.download('punkt_tab')
seed_everything(42, workers=True)


torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    print("\n[TRAINER] Starting...\n")

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-d', '--dataset', type=str, default='CoAuthor', choices=["CoAuthor", "SeqXGPT-Bench"], help='Dataset to use')
    arg_parser.add_argument('-ra_d', '--raw_data_path', type=str, help='Path to the raw data')
    arg_parser.add_argument('-pa_d', '--parsed_data_path', type=str, help='Path to the parsed data')
    arg_parser.add_argument('-pr_d', '--preprocessed_data_path', type=str, help='Path to the preprocessed data')
    arg_parser.add_argument('-r', '--recreate', action='store_true', help='Recreate the processed data')
    arg_parser.add_argument('-t', '--type', type=str, default='all', choices=["creative", "argumentative", "all"], help='Type of train data to use')
    arg_parser.add_argument('-tr_s', '--train_sources', nargs='+', help="Sources to use for training, separated by space. Available sources: SeqXGPT-Bench('gpt2', 'gpt3', 'gptj', 'gptneo', 'human', 'llama')")
    arg_parser.add_argument('-te_s', '--test_sources', nargs='+', help="Sources to use for test, separated by space. Available sources: SeqXGPT-Bench('gpt2', 'gpt3', 'gptj', 'gptneo', 'human', 'llama')")
    arg_parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    arg_parser.add_argument('-e', '--max_epochs', type=int, default=10, help='Maximum number of epochs')
    arg_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    arg_parser.add_argument('-dr', '--dropout', type=float, default=0.3, help='Dropout rate')
    arg_parser.add_argument('-a', '--alpha', type=float, default=1.0, help='Alpha for the loss')
    arg_parser.add_argument('-s_p', '--model_save_path', type=str, default='weights/detector', help='Path to save the model')
    arg_parser.add_argument('-l_p', '--log_path', type=str, default='logs/detector', help='Path to save the logs')
    arg_parser.add_argument('-gc', '--global_coherence', action='store_true', help='Use global coherence feature')
    arg_parser.add_argument('-lc', '--local_coherence', action='store_true', help='Use local coherence feature')
    arg_parser.add_argument('-gl', '--global_lexical', action='store_true', help='Use global lexical feature')
    arg_parser.add_argument('-ll', '--local_lexical', action='store_true', help='Use local lexical feature')
    arg_parser.add_argument('-te', '--test', action='store_true', help='Test run')

    args = arg_parser.parse_args()

    global_coherence_feature = args.global_coherence if args.global_coherence else False
    local_coherence_feature = args.local_coherence if args.local_coherence else False
    global_lexical_feature = args.global_lexical if args.global_lexical else False
    local_lexical_feature = args.local_lexical if args.local_lexical else False

    test_run = args.test if args.test else False

    if args.dataset == "CoAuthor":
        wandb_project = "[SL] [CoAuthor] [GL-CLiC]"

        if args.raw_data_path is None:
            raw_data_path = "dataset/CoAuthor/raw/coauthor.csv"
        else:
            raw_data_path = args.raw_data_path

        if args.parsed_data_path is None:
            parsed_data_path = "dataset/CoAuthor/parsed/coauthor.pkl"
        else:
            parsed_data_path = args.parsed_data_path

        if args.preprocessed_data_path is None:
            preprocessed_data_path = "dataset/CoAuthor/preprocessed/coauthor.pkl"
        else:
            preprocessed_data_path = args.preprocessed_data_path

    elif args.dataset == "SeqXGPT-Bench":
        wandb_project = "[SL] [SeqXGPT-Bench] [GL-CLiC]"

        if args.raw_data_path is None:
            raw_data_path = "dataset/SeqXGPT-Bench/raw"
        else:
            raw_data_path = args.raw_data_path

        if args.parsed_data_path is None:
            parsed_data_path = "dataset/SeqXGPT-Bench/parsed/seqxgpt-bench.pkl"
        else:
            parsed_data_path = args.parsed_data_path

        if args.preprocessed_data_path is None:
            preprocessed_data_path = "dataset/SeqXGPT-Bench/preprocessed/seqxgpt-bench.h5"

    trainer_config = GLCLiCTrainerConfig(
        test=test_run,
        dataset=args.dataset,
        train_sources=args.train_sources,
        test_sources=args.test_sources,
        wandb_project=wandb_project,
        raw_data_path=raw_data_path,
        parsed_data_path=parsed_data_path,
        preprocessed_data_path=preprocessed_data_path,
        recreate=args.recreate,
        experiment_type=args.type,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        alpha=args.alpha,
        model_save_path=args.model_save_path,
        log_path=args.log_path,
        global_coherence=global_coherence_feature,
        local_coherence=local_coherence_feature,
        global_lexical=global_lexical_feature,
        local_lexical=local_lexical_feature
    )

    print("\n[TRAINER] Initializing trainer...\n")
    trainer = GLCLiCTrainer(trainer_config)

    trainer.train()

    trainer.test()

    trainer.finish()
