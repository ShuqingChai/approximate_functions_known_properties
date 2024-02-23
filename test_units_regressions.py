import unittest
import numpy as np
from samplers import generate_task_random, generate_feature_random, equation, generate_data, OnlineSampler
from algorithms import Perturbations

np.random.seed(42)

class TestSamplers(unittest.TestCase):

    def test_generate_task_random(self):
        Theta, sigma = generate_task_random()
        self.assertEqual(len(Theta), 4)
        self.assertTrue(0 <= sigma <= 2)

    def test_generate_feature_random(self):
        xy_bd = [0, 1, 0, 1]
        feature = generate_feature_random(xy_bd)
        self.assertEqual(len(feature), 5)
        self.assertTrue(0 <= feature[0] <= 1)
        self.assertTrue(0 <= feature[1] <= 1)

    def test_equation(self):
        model = "linear"
        features = np.array([0.5, 0.5, 0, 0, 2])
        xy_bd = [0, 1, 0, 1]
        Theta = np.array([1, 1, 1, 1])
        response = equation(model, features, xy_bd, Theta, False)
        self.assertEqual(response, 2.75)

    def test_generate_data(self):
        model = "linear"
        num_data = 10
        xy_bd = [0, 1, 0, 1]
        Theta = np.array([1, 1, 1, 1])
        sigma = 0.5
        features, responses = generate_data(model, num_data, xy_bd, Theta, sigma, False)
        self.assertEqual(features.shape, (num_data, 5))
        self.assertEqual(responses.shape, (num_data, 1))

    ''' Unit tests for OnlineSampler class'''
    def setUp(self):
        self.model = "linear"
        self.num_data = 10
        self.xy_bd = [0, 1, 0, 1]
        self.sampler = OnlineSampler(self.model, self.num_data, self.xy_bd)

    def test_sample(self):
        features, responses = self.sampler.sample()
        self.assertEqual(features.shape, (self.num_data, 5))
        self.assertEqual(responses.shape, (self.num_data, 1))
        # print("features: ", features)
        # print("responses: ", responses)
        # print("Theta: ", self.sampler.Theta)
        # print("sigma: ", self.sampler.sigma)

    def test_get_fresh_values(self):
        params = np.array([0.5, 0.5, 0, 0, 2])
        self.sampler.sigma = 0.5
        self.sampler.Theta = np.array([1, 1, 1, 1])
        response = self.sampler.get_fresh_values(params)
        self.assertEqual(response, 2.75)

    def test_loss(self):
        epsilons = np.array([[0.1, 0.2, 0.3, 0.4, 5]])
        c = 0.5
        antithetic = False
        detailed_losses = self.sampler.loss(epsilons, c, antithetic)
        self.assertEqual(detailed_losses.shape, (epsilons.shape[0], 2))

    def test_pollute_observation(self):
        self.sampler.noise = "fixed_variance"
        self.sampler.sigma = 2
        data = np.array([1, 2, 3, 4, 5])
        polluted_data = self.sampler.pollute_observation(data)
        self.assertNotEqual(sum(np.square(polluted_data- data)),0)
        
class TestPerturbations(unittest.TestCase):

    def test_init_gs(self):
        perturbations = Perturbations('gs', 10, 1)
        self.assertEqual(perturbations.algo_name, 'gs')
        self.assertEqual(perturbations.dim, 10)
        self.assertEqual(perturbations.L, 1)
        self.assertEqual(perturbations.dist, 'gaussian')
        self.assertEqual(perturbations.sigma, 1)

    def test_generate_gaussian(self):
        perturbations = Perturbations('gs', 10, 1)
        result = perturbations.generate(5)
        self.assertEqual(result.shape, (5, 10))

if __name__ == '__main__':
    unittest.main()
    
