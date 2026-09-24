import unittest

import torch as th

from macflow.theta_metrics import ThetaMetrics


class ThetaMetricsTest(unittest.TestCase):
    def test_two_dimensional_trajectory(self):
        tracker = ThetaMetrics()
        positions = [(5, 5), (7, 5), (10, 5), (10, 9), (10, 7), (5, 7), (5, 5)]
        expected_steps = [0, 2, 3, 4, 2, 5, 2]
        expected_cosines = [0, 0, 1, 0, -1, 0, 0]
        expected_distances = [0, 2, 5, 41 ** 0.5, 29 ** 0.5, 2, 0]
        for i, position in enumerate(positions):
            metrics = tracker.measure([th.tensor(position, dtype=th.float64)])
            self.assertAlmostEqual(metrics["theta_step_norm"], expected_steps[i])
            self.assertAlmostEqual(metrics["theta_step_cos"], expected_cosines[i])
            self.assertAlmostEqual(metrics["theta_disp_norm"], expected_distances[i])
            self.assertEqual(metrics["theta_step_cos_valid"], float(i >= 2))
        th.testing.assert_close(tracker.reference, th.tensor([5., 5.], dtype=th.float64))

    def test_checkpoint_preserves_reference_and_direction(self):
        tracker = ThetaMetrics()
        tracker.measure([th.tensor([5., 5.])])
        tracker.measure([th.tensor([7., 5.])])
        restored = ThetaMetrics()
        restored.load_state_dict(tracker.state_dict())
        current = [th.tensor([10., 5.])]
        self.assertEqual(tracker.measure(current), restored.measure(current))
        self.assertEqual(restored.measure([th.tensor([10., 9.])])["theta_disp_norm"], th.tensor(41.).sqrt().item())

    def test_zero_step_has_no_direction(self):
        tracker = ThetaMetrics()
        for position in ([0., 0.], [1., 0.], [1., 0.], [2., 0.]):
            metrics = tracker.measure([th.tensor(position)])
            self.assertEqual(metrics["theta_step_cos_valid"], 0)
            self.assertEqual(metrics["theta_step_cos"], 0)

    def test_measurement_does_not_modify_parameters_gradients_or_rng(self):
        parameter = th.nn.Parameter(th.tensor([3., 4.]))
        parameter.grad = th.tensor([1., 2.])
        before = parameter.detach().clone()
        gradient = parameter.grad.clone()
        rng = th.get_rng_state().clone()
        ThetaMetrics().measure([parameter])
        th.testing.assert_close(parameter, before)
        th.testing.assert_close(parameter.grad, gradient)
        self.assertTrue(th.equal(th.get_rng_state(), rng))


if __name__ == "__main__":
    unittest.main()
