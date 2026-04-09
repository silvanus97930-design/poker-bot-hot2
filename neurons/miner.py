"""Reference Poker44 miner with chunk-level GRU+Transformer inference."""

# from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Tuple

import bittensor as bt
import torch
import numpy as np

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse
from poker_bot_detection.models.gru_model import GRUTransformerClassifier
import poker_bot_detection.config as det_config
from poker_bot_detection.utils.features import encode_hand
from poker_bot_detection.utils.dataset import load_feature_norm, apply_feature_normalization


class Miner(BaseMinerNeuron):
    """GRU+Transformer miner that returns one bot-risk score per chunk."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 GRU+Transformer Poker44 Miner started")
        repo_root = Path(__file__).resolve().parents[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = repo_root / "best_model.pt"
        self.norm_path = repo_root / "poker_bot_detection" / "feature_norm.pt"
        self.decision_threshold = float(
            os.getenv(
                "MODEL_DECISION_THRESHOLD",
                getattr(det_config, "INFERENCE_THRESHOLD", 0.5),
            )
        )
        self.model = self._load_model()
        self.feature_mean, self.feature_std = self._load_normalization()
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "poker44-gru-transformer-strong-seed44",
                "model_version": "2",
                "framework": "pytorch-gru-transformer",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": "GRU+Transformer miner using strong seed44 checkpoint and train-fit normalization.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on strong seed44 public benchmark chunks using GRU+Transformer sequence model."
                ),
                "training_data_sources": ["data/public_miner_benchmark_strong_seed44.json.gz"],
                "private_data_attestation": (
                    "This miner is trained only on public benchmark data, not validator-private human data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            f"Inference threshold={self.decision_threshold:.3f} "
            "(override with MODEL_DECISION_THRESHOLD env var)"
        )
        bt.logging.info(
            "Miner prep tooling available | "
            f"benchmark_doc={repo_root / 'docs' / 'public-benchmark.md'} "
            f"miner_doc={repo_root / 'docs' / 'miner.md'} "
            f"anti_leakage_doc={repo_root / 'docs' / 'anti-leakage.md'}"
        )
        bt.logging.info(
            "Public benchmark command: "
            "python scripts/publish/publish_public_benchmark.py --skip-wandb"
        )
        bt.logging.info(
            "Purpose: train, validate and refine miner models against the public benchmark "
            "while Poker44 moves toward more dynamic evaluation."
        )

    def _load_model(self) -> GRUTransformerClassifier:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {self.model_path}")
        model = GRUTransformerClassifier(
            input_dim=det_config.INPUT_DIM,
            hidden_dim=det_config.HIDDEN_DIM,
            gru_layers=det_config.NUM_LAYERS,
            tf_layers=getattr(det_config, "TF_LAYERS", 2),
            nhead=getattr(det_config, "NHEAD", 4),
            ff_mult=getattr(det_config, "FF_MULT", 4),
            dropout=det_config.DROPOUT,
            max_len=getattr(det_config, "MAX_SEQ_LEN", 2048),
            bidirectional_gru=getattr(det_config, "BIDIRECTIONAL_GRU", True),
            use_attention_pool=getattr(det_config, "USE_ATTENTION_POOL", True),
        ).to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        bt.logging.info(f"Loaded GRU+Transformer checkpoint from {self.model_path}")
        return model

    def _load_normalization(self):
        if not self.norm_path.exists():
            raise FileNotFoundError(f"Missing feature norm file: {self.norm_path}")
        mean, std, input_dim = load_feature_norm(self.norm_path)
        if input_dim is not None and int(input_dim) != int(det_config.INPUT_DIM):
            raise ValueError(
                f"Normalization input_dim mismatch: norm={input_dim}, config={det_config.INPUT_DIM}"
            )
        bt.logging.info(f"Loaded feature normalization from {self.norm_path}")
        return mean.to(self.device), std.to(self.device)

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one GRU+Transformer risk score per chunk."""
        chunks = synapse.chunks or []
        scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [s >= self.decision_threshold for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner Predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with model risks.")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def score_chunk(self, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        seq = np.array([encode_hand(hand) for hand in chunk], dtype=np.float32)
        x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=self.device)
        x = apply_feature_normalization(
            x,
            self.feature_mean,
            self.feature_std,
            det_config.FEATURE_NORM_EPS,
        )
        with torch.no_grad():
            logits = self.model(x, lengths=lengths)
            prob = torch.sigmoid(logits).item()
        return round(self._clamp01(prob), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
