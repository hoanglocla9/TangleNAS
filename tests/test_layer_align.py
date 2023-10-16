import unittest
import torch
import math
from toy_search_spaces.cell_topology.model_search import ToyCellSearchSpace

class TestLayerAlignmentScore(unittest.TestCase):

    def _test_score_not_nan(self, search_model, criterion):
        x = torch.randn(2, 1, 28, 28)
        targets = torch.randint(0, 10, (2,))

        output, logits = search_model(x)
        loss = criterion(logits, targets)
        loss.backward()

        score = search_model.compute_layer_align_score("reduce")
        print(score)
        assert not math.isnan(score), "Score is NaN"


    def test_toy_search_space_layer_align_darts_v1_we(self):
        criterion = torch.nn.CrossEntropyLoss()
        search_model = ToyCellSearchSpace(
            optimizer_type="darts_v1",
            criterion=criterion,
            num_classes=10,
            entangle_weights=True,
            use_we_v2=True
        )

        self._test_score_not_nan(search_model, criterion)


    def test_toy_search_space_layer_align_darts_v1_ws(self):
        criterion = torch.nn.CrossEntropyLoss()

        search_model = ToyCellSearchSpace(
            optimizer_type="darts_v1",
            criterion=criterion,
            num_classes=10,
            entangle_weights=False,
            use_we_v2=False
        )

        self._test_score_not_nan(search_model, criterion)

    def test_toy_search_space_layer_align_drnas_we(self):
        criterion = torch.nn.CrossEntropyLoss()
        search_model = ToyCellSearchSpace(
            optimizer_type="drnas",
            criterion=criterion,
            num_classes=10,
            entangle_weights=True,
            use_we_v2=True
        )

        self._test_score_not_nan(search_model, criterion)

    def test_toy_search_space_layer_align_drnas_ws(self):
        criterion = torch.nn.CrossEntropyLoss()
        search_model = ToyCellSearchSpace(
            optimizer_type="drnas",
            criterion=criterion,
            num_classes=10,
            entangle_weights=False,
            use_we_v2=False
        )

        self._test_score_not_nan(search_model, criterion)

    def test_toy_search_space_layer_align_gdas_we(self):
        criterion = torch.nn.CrossEntropyLoss()
        search_model = ToyCellSearchSpace(
            optimizer_type="gdas",
            criterion=criterion,
            num_classes=10,
            entangle_weights=True,
            use_we_v2=True
        )

        search_model.sampler.set_taus(tau_min=0.1, tau_max=1.0)
        search_model.sampler.set_total_epochs(50)
        search_model.sampler.before_epoch()
        self._test_score_not_nan(search_model, criterion)

    def test_toy_search_space_layer_align_gdas_ws(self):
        criterion = torch.nn.CrossEntropyLoss()
        search_model = ToyCellSearchSpace(
            optimizer_type="gdas",
            criterion=criterion,
            num_classes=10,
            entangle_weights=False,
            use_we_v2=False
        )

        search_model.sampler.set_taus(tau_min=0.1, tau_max=1.0)
        search_model.sampler.set_total_epochs(50)
        search_model.sampler.before_epoch()
        self._test_score_not_nan(search_model, criterion)


if __name__ == '__main__':
    unittest.main()
