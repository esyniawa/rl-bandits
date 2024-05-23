import os.path

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class NArmedBandit:
    def __init__(self, n_arms, rewards: list | tuple | None = None, seed: None | int = None):
        self.n_arms = n_arms
        if rewards is None:
            self.rewards = np.random.RandomState(seed).random(self.n_arms)
        else:
            assert len(rewards) == n_arms, 'The rewards vector must have the same length as the number of arms'
            self.rewards = rewards

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
                 seed: int | None = None,
                 animate: bool = False,
                 **kwargs):

        estimates = np.zeros(self.n_arms)
        counts = np.zeros(self.n_arms)
        rewards = []
        np.random.seed(seed)

        if animate:
            estimates_trajectory = np.zeros((num_iterations, self.n_arms))

        for i in range(num_iterations):

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

            if animate:
                estimates_trajectory[i, :] = estimates

        if animate:

            fig, ax = plt.subplots()

            # plotting
            x_axis = np.arange(self.n_arms) + 1
            ax.bar(x=x_axis, height=self.rewards, width=0.5, zorder=1, alpha=0.3, color='b')
            l = ax.bar(x=x_axis, height=estimates_trajectory[0], width=0.5, zorder=2, alpha=0.3, color='r')

            ax.set_ylabel('Estimates')
            ax.set_xlabel('Arms', loc='right')

            def update_animate(t):
                for j, bar in enumerate(l):
                    bar.set_height(estimates_trajectory[t, j])
                return l


            ani = animation.FuncAnimation(fig, update_animate, frames=np.arange(0, num_iterations-1))
            writer = animation.FFMpegWriter(fps=4)

            if not os.path.exists("videos/"):
                os.mkdir("videos/")
            ani.save('videos/trajectory_' + strategy + '.mp4', writer=writer)
            plt.close(fig)

        else:
            return np.mean(rewards), np.mean(rewards) / np.amax(self.rewards)

