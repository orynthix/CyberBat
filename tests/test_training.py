import torch

from train import finetune_attack, pretrain_dynamics, train_from_file


class IdentityEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def encode(self, values):
        return values * self.scale

    def decode(self, latent):
        return latent


class LastStateTransition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.input_sequence_length = None

    def forward(self, history):
        self.input_sequence_length = history.shape[1]
        return history[:, -1] + self.offset


def test_dynamics_pretraining_predicts_adjacent_next_window() -> None:
    encoder = IdentityEncoder()
    world_model = LastStateTransition()
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(world_model.parameters()),
        lr=0.0,
    )
    history = torch.tensor([[[1.0], [2.0]]])
    future_states = torch.tensor([[[3.0], [4.0]]])

    losses = pretrain_dynamics(
        encoder,
        world_model,
        [(history, future_states)],
        optimizer,
        epochs=1,
    )

    assert world_model.input_sequence_length == 2
    assert losses == [1.0]


def test_attack_finetuning_aligns_each_rollout_prefix_with_its_future_label() -> None:
    class FixedRollout(torch.nn.Module):
        def forward_rollout(self, history, steps):
            values = torch.arange(1, steps + 1, dtype=history.dtype, device=history.device)
            return values.view(1, steps, 1).expand(history.shape[0], -1, -1)

    class RecordingHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def loss(self, trajectory, risk_target, stage_target):
            self.calls.append((trajectory.shape[1], risk_target.clone(), stage_target.clone()))
            return self.weight * 0

    world_model = FixedRollout()
    attack_head = RecordingHead()
    optimizer = torch.optim.SGD(attack_head.parameters(), lr=0.0)
    risk_targets = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    stage_targets = torch.tensor([[0, 1, 0], [2, 0, 3]])

    losses = finetune_attack(
        world_model,
        attack_head,
        [(torch.zeros((2, 2, 1)), risk_targets, stage_targets)],
        optimizer,
        epochs=1,
        rollout_steps=3,
    )

    assert losses == [0.0]
    assert [call[0] for call in attack_head.calls] == [1, 2, 3]
    assert [call[1].tolist() for call in attack_head.calls] == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    assert [call[2].tolist() for call in attack_head.calls] == [[0, 2], [1, 0], [0, 3]]


def test_training_saves_horizon_aligned_supervision_metadata(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    import train
    from src.config import DEFAULT_CONFIG, ModelConfig, WindowConfig

    config = replace(
        DEFAULT_CONFIG,
        windows=WindowConfig(sequence_length=2, rollout_steps=2),
        model=ModelConfig(
            latent_dim=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        ),
    )
    monkeypatch.setattr(train, "DEFAULT_CONFIG", config)
    source = tmp_path / "traffic.csv"
    source.write_text(
        "timestamp,bytes,packets,stage\n"
        "0,10,1,0\n"
        "5,20,1,1\n"
        "10,30,1,2\n"
        "15,40,1,3\n",
        encoding="utf-8",
    )

    result = train_from_file(
        source,
        tmp_path / "model",
        dynamics_epochs=1,
        attack_epochs=1,
        batch_size=1,
    )
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)

    assert checkpoint["attack_supervision"] == "horizon_aligned_rollout_prefixes_v1"
    assert len(checkpoint["attack_loss"]) == 1
