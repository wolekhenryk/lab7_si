import numpy as np
from rl_base import Agent, Action, State
import os


class QAgent(Agent):

    def __init__(self, n_states, n_actions,
                 name='QAgent', initial_q_value=0.0, q_table=None):
        super().__init__(name)

        # hyperparams
        # TODO ustaw te parametry na sensowne wartości
        self.lr = 0.05                   # współczynnik uczenia (learning rate)
        self.gamma = 0.75                # współczynnik dyskontowania
        self.epsilon = 1.0              # epsilon (p-wo akcji losowej)
        self.eps_decrement = 0.001     # wartość, o którą zmniejsza się epsilon po każdym kroku
        self.eps_min = 0.01             # końcowa wartość epsilon, poniżej którego już nie jest zmniejszane

        self.action_space = [i for i in range(n_actions)]
        self.n_states = n_states
        self.q_table = q_table if q_table is not None else self.init_q_table(initial_q_value)

    def init_q_table(self, initial_q_value=0.) -> np.ndarray:
        # TODO - utwórz tablicę wartości Q o rozmiarze [n_states, n_actions] wypełnioną początkowo wartościami
        #  initial_q_value
        return np.full((self.n_states, len(self.action_space)), initial_q_value)

    def update_action_policy(self) -> None:
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_decrement
        else:
            self.epsilon = self.eps_min

    def choose_action(self, state: State) -> Action:

        assert 0 <= state < self.n_states, \
            f"Bad state_idx. Has to be int between 0 and {self.n_states}"

        # TODO - zaimplementuj strategię eps-zachłanną
        action = np.argmax(self.q_table[state]) \
            if np.random.uniform() > self.epsilon \
            else np.random.choice(self.action_space)
        return Action(action)

    def learn(self, state: State, action: Action, reward: float, new_state: State, done: bool) -> None:
        q_value = self.q_table[state][action]
        new_q_value = q_value + self.lr * (reward + self.gamma * np.max(self.q_table[new_state]) - q_value)
        self.q_table[state][action] = new_q_value

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    def get_instruction_string(self):
        return [f"Linearly decreasing eps-greedy: eps={self.epsilon:0.4f}"]
