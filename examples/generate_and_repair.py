"""Exercise an untrained cCorrGAN and repair; no fidelity claim."""

import torch
from corrbench.geometry import nearest_correlation_torch

from corrgan import ConditionalCorrGANGenerator

torch.manual_seed(7)
torch.set_num_threads(2)
model = ConditionalCorrGANGenerator(channels=8, continuous_condition=False).eval()
with torch.no_grad():
    raw = model(torch.randn(3, 80), torch.arange(3))
    upper = torch.triu(raw, diagonal=1)
    candidate = upper + upper.transpose(-1, -2) + torch.eye(80)
    repaired, info = nearest_correlation_torch(
        candidate.double(), tolerance=1e-6, max_iterations=1000, return_info=True
    )
assert bool(torch.all(info["converged"]))
print("Untrained API smoke:", tuple(repaired.shape), "repair converged")
