from dataclasses import dataclass

WEAK_MODE = False

@dataclass
class ProjectConfig:
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    dataset_path: str = "/content/drive/MyDrive/MultiDomain_Instruction_50k"
    output_dir: str = "/content/drive/MyDrive/Qwen2-0.5B-SFT-MultiDomain"
    max_seq_length: int = 512
    seed: int = 42