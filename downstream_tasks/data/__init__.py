"""Load downstream datasets and embeddings."""
from .loaders import EmbeddingLoader
from .datasets import ABMILDataset, collate_abmil
from .task_loaders import get_task_loader
