"""
    Perturbation-based attacks.
"""
from typing import Set

from mimir.attacks.all_attacks import Attack
from mimir.models import Model, PerturbationModel
from mimir.config import ExperimentConfig


class PerturbationAttack(Attack):

    def __init__(
        self, config: ExperimentConfig,
        model: Model,
        perturbation_models: Set[PerturbationModel],
    ):
        super().__init__(config, model, ref_model=None, perturb_models=perturbation_models)

    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Perturbation-based attack score.
        """
        loss = kwargs.get('loss', None)
        if loss is None:
            loss = self.target_model.get_ll(document, probs=probs, tokens=tokens)
        perturb_loss = 0.0
        for perturb_model in self.perturb_models:
            perturb_loss += perturb_model.get_ll(document, probs=None, tokens=tokens)
        perturb_loss /= len(self.perturb_models)
        # 越小越可能是成员样本
        return loss - perturb_loss
