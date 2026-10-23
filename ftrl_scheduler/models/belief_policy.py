"""
belief_policy.py
----------------
LSTM-based belief state estimator + custom PPO policy (FT-RLS agent).

Architecture
------------
Input  : noisy obs history (LSTM processes step-by-step)
LSTM   : 2 layers, hidden_size=256
Linear : 256 -> belief_dim (128)
PPO    : acts on belief vector b_t (not raw obs)
"""

import torch
import torch.nn as nn
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces


# ── Belief Estimator ─────────────────────────────────────────────────────────

class LSTMBeliefEstimator(nn.Module):
    """
    2-layer LSTM that maps a sequence of noisy observations to a
    compact belief vector b_t ∈ R^{belief_dim}.

    Input at each step : noisy_obs || missingness_mask  (dim = obs_dim * 2)
    Output             : belief vector b_t              (dim = belief_dim)
    """

    def __init__(self, input_dim, hidden_size=256, belief_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.belief_dim  = belief_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, belief_dim),
            nn.LayerNorm(belief_dim),
            nn.Tanh(),
        )

    def forward(self, obs_seq, hidden=None):
        """
        obs_seq : (batch, seq_len, input_dim)  OR  (batch, input_dim) for single step
        Returns : belief (batch, belief_dim), (h_n, c_n)
        """
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)        # add seq_len=1 dim
        lstm_out, hidden = self.lstm(obs_seq, hidden)
        last_hidden      = lstm_out[:, -1, :]    # take last timestep
        belief           = self.proj(last_hidden)
        return belief, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


# ── Feature Extractor (plugs into SB3) ───────────────────────────────────────

class BeliefFeatureExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible feature extractor that wraps the LSTM belief estimator.
    SB3 calls this once per environment step with the current observation.
    The LSTM hidden state is maintained across steps via a stateful buffer.
    """

    def __init__(self, observation_space: spaces.Box,
                 hidden_size: int = 256,
                 belief_dim: int  = 128,
                 n_layers: int    = 2):
        super().__init__(observation_space, features_dim=belief_dim)
        input_dim = observation_space.shape[0]
        self.estimator = LSTMBeliefEstimator(
            input_dim=input_dim,
            hidden_size=hidden_size,
            belief_dim=belief_dim,
            n_layers=n_layers,
        )
        # stateful hidden — reset when env resets (SB3 calls reset_hidden externally)
        self._hidden = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """observations : (batch, obs_dim)  — SB3 processes one step at a time."""
        device = observations.device
        batch  = observations.shape[0]

        if self._hidden is None or self._hidden[0].shape[1] != batch:
            self._hidden = self.estimator.init_hidden(batch, device)

        belief, self._hidden = self.estimator(observations, self._hidden)
        # detach hidden to prevent BPTT through full episode (PPO processes fixed rollouts)
        self._hidden = (self._hidden[0].detach(), self._hidden[1].detach())
        return belief

    def reset_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        self._hidden = self.estimator.init_hidden(batch_size, device)


# ── Custom PPO Policy ─────────────────────────────────────────────────────────

class BeliefActorCriticPolicy(MaskableActorCriticPolicy):
    """
    PPO Actor-Critic policy that uses the LSTM belief estimator
    as its feature extractor.
    """

    def __init__(self, *args,
                 hidden_size: int = 256,
                 belief_dim:  int = 128,
                 n_lstm_layers: int = 2,
                 **kwargs):
        self._hidden_size   = hidden_size
        self._belief_dim    = belief_dim
        self._n_lstm_layers = n_lstm_layers
        super().__init__(*args,
                         features_extractor_class=BeliefFeatureExtractor,
                         features_extractor_kwargs=dict(
                             hidden_size=hidden_size,
                             belief_dim=belief_dim,
                             n_layers=n_lstm_layers,
                         ),
                         net_arch=dict(pi=[128, 64], vf=[128, 64]),
                         **kwargs)

    def reset_belief(self, batch_size=1):
        """Reset LSTM hidden state — called on env reset."""
        self.features_extractor.reset_hidden(batch_size)


class MLPPolicy(MaskableActorCriticPolicy):
    """
    Standard MLP policy that acts directly on noisy observations.
    Used as the 'noisy baseline' (PPO with raw noisy obs, no belief state).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                         **kwargs)
