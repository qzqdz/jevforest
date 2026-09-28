"""Product SDK.

`decide()` is not implemented (P4). Use ForestAcquisitionPolicy / eval-afa
for tabular AFA, or JevDecisionsClient for typed OpenRouter questions.
"""

from jevforest.sdk.jev import JevDecisionsClient, JevDecisionsError

__all__ = ["decide", "JevDecisionsClient", "JevDecisionsError"]


def decide(*_args, **_kwargs):  # noqa: ANN001, ANN003
    raise NotImplementedError(
        "jevforest.sdk.decide is not implemented. "
        "Use ForestAcquisitionPolicy / eval-afa, or JevDecisionsClient."
    )
