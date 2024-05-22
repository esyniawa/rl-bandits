import numpy as np


class NArmedBandit:
    def __init__(self, n_arms, mu: float = 0, sigma: float = 1, seed: None | int = None):
        self.n_arms = n_arms
        self.rewards = np.random.RandomState(seed).normal(mu, sigma, n_arms)

    def pull(self, arm):
        return np.random.rand() < self.rewards[arm]

    def softmax(self, estimates: np.ndarray, temp: float = 1.):
        values = np.exp(estimates / temp)
        return values / np.sum(values)

    def reinforcement_comparison(self, estimates: np.ndarray, baseline: float = 0.):
        values = estimates - baseline
        preferences = (values > 0).astype(int)
        if np.sum(preferences) == 0:
            return np.ones(self.n_arms) / self.n_arms
        else:
            return preferences / np.sum(preferences)

    def gradient_bandit(self, estimates: np.ndarray, baseline: float = 0., step_size: float = 0.1):
        preferences = self.softmax(estimates)
        selected_arm = np.random.choice(self.n_arms, p=preferences)
        reward = self.pull(selected_arm)
        for arm in range(self.n_arms):
            if arm == selected_arm:
                estimates[arm] += step_size * (reward - baseline) * (1 - preferences[arm])
            else:
                estimates[arm] -= step_size * (reward - baseline) * preferences[arm]
        return estimates

    def ucb(self, estimates: np.ndarray, counts: np.ndarray, c=2):
        total_counts = np.sum(counts)
        if total_counts == 0:
            return np.random.randint(self.n_arms)
        else:
            values = estimates + c * np.sqrt(np.log(total_counts) / (counts + 1e-8))
            return np.argmax(values)

    def evaluate(self,
                 strategy: str,
                 num_iterations: int = 1000,
                 seed=None, **kwargs):

        estimates = np.zeros(self.n_arms)
        counts = np.zeros(self.n_arms)
        rewards = []
        np.random.seed(seed)
        for _ in range(num_iterations):

            # switch
            if strategy == 'softmax':
                preferences = self.softmax(estimates, **kwargs)
                selected_arm = np.random.choice(self.n_arms, p=preferences)
            elif strategy == 'reinforcement_comparison':
                preferences = self.reinforcement_comparison(estimates)
                selected_arm = np.random.choice(self.n_arms, p=preferences / np.sum(preferences))
            elif strategy == 'gradient_bandit':
                estimates = self.gradient_bandit(estimates, **kwargs)
                preferences = self.softmax(estimates)
                selected_arm = np.random.choice(self.n_arms, p=preferences)
            elif strategy == 'ucb':
                selected_arm = self.ucb(estimates, counts, **kwargs)
            else:
                raise AssertionError('You must choose a implemented strategy!')

            reward = self.pull(selected_arm)
            rewards.append(reward)
            counts[selected_arm] += 1
            estimates[selected_arm] += (reward - estimates[selected_arm]) / counts[selected_arm]

        return np.mean(rewards), np.mean(rewards) / np.amax(self.rewards)

