"""Ensures GFLOP logging behaves."""

import torch
from torch import nn
from torch.utils import data

from yoyodyne import models, trainers


class DummyTarget:
    """Mocks the interface of target sequence wrapper."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


class DummyBatch:
    """Mocks the interface of data.Batch."""

    def __init__(self, source: torch.Tensor, target: torch.Tensor):
        self.source = source
        self.target = DummyTarget(target)

    def __len__(self) -> int:
        return self.source.size(0)


class DummyDataset(data.Dataset):
    """Generates deterministic toy samples."""

    def __init__(self):
        self.samples = [
            (torch.randn(5, 10), torch.randint(0, 10, (5,))) for _ in range(8)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate(batch_list: list[tuple[torch.Tensor, torch.Tensor]]) -> DummyBatch:
    """Collates raw tuples into a mock data.Batch object."""
    sources = torch.stack([s for s, t in batch_list])
    targets = torch.stack([t for s, t in batch_list])
    return DummyBatch(sources, targets)


class SimpleLinearModel(models.BaseModel):
    """Minimal implementation to verify base profiling hooks."""

    def __init__(self, **kwargs):
        # Target_vocab_size must match for CrossEntropyLoss.
        super().__init__(vocab_size=10, target_vocab_size=10, **kwargs)
        self.layer = nn.Linear(10, 10, bias=False)
        self.decoder = nn.Identity()

    def get_decoder(self) -> nn.Module:
        return nn.Identity()

    @staticmethod
    def init_embeddings(
        num_embeddings: int, embedding_size: int
    ) -> nn.Embedding:
        return nn.Embedding(num_embeddings, embedding_size)

    def forward(self, batch: DummyBatch) -> torch.Tensor:
        return self.layer(batch.source).transpose(1, 2)


def test_gflop_counter_logging():
    dataset = DummyDataset()
    dataloader = data.DataLoader(dataset, batch_size=4, collate_fn=collate)
    model = SimpleLinearModel(compute_gflop=True)
    trainer = trainers.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=True,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    assert model.flop is None
    assert not model.has_flop
    assert "train_gflop" in trainer.logged_metrics
    assert trainer.logged_metrics["train_gflop"] > 0.0
