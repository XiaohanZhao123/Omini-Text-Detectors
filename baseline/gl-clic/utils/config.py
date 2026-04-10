from dataclasses import dataclass


@dataclass
class GLCLiCModelConfig:
    seed: int
    dataset: str = None
    train_sources: list = None
    test_sources: list = None
    train_data_type: str = None
    global_coherence: bool = False
    local_coherence: bool = False
    global_lexical: bool = False
    local_lexical: bool = False
    use_gpu: bool = False
    lr: float = 1e-4
    dropout: float = 0.3
    alpha: float = 1.0

    def print_config(self):
        print("\n[CONFIG] GLCLiC Model")
        print(f"[CONFIG] Seed: {self.seed}")
        print(f"[CONFIG] Dataset: {self.dataset}")
        if self.dataset == "SimLLM":
            print(f"[CONFIG] Train Sources: {self.train_sources}")
            print(f"[CONFIG] Test Sources: {self.test_sources}")
        if self.dataset == "CoAuthor":
            print(f"[CONFIG] Train Data Type: {self.train_data_type}")
        print(f"[CONFIG] Global Coherence Feature: {self.global_coherence}")
        print(f"[CONFIG] Local Coherence Feature: {self.local_coherence}")
        print(f"[CONFIG] Global Lexical Feature: {self.global_lexical}")
        print(f"[CONFIG] Local Lexical Feature: {self.local_lexical}")
        print(f"[CONFIG] Use GPU: {self.use_gpu}")
        print(f"[CONFIG] Learning Rate: {self.lr}")
        print(f"[CONFIG] Dropout: {self.dropout}")
        print(f"[CONFIG] Alpha: {self.alpha}")
        print()
