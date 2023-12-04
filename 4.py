import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, step_size=0.01, discount_factor=0.9, epsilon=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))  # Q-Table 초기화
        self.step_size = step_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def select_action(self, state):
        # Epsilon-Greedy 정책을 사용하여 액션 선택
        if np.random.rand() < self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.choice(self.num_actions)

    def update_q_table(self, state, action, reward, next_state):
        # 벨만 최적 방정식을 사용한 Q-Table 업데이트
        q_1 = self.q_table[state][action]
        q_2 = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.step_size * (q_2 - q_1)

# 간단한 환경 예제
num_states = 16
num_actions = 4

# Q-Learning 에이전트 초기화
agent = QLearningAgent(num_states=num_states, num_actions=num_actions)

# 학습 예제
num_episodes = 1000

for episode in range(num_episodes):
    state = 0  # 시작 상태
    total_reward = 0

    while state != num_states - 1:  # 목표 상태에 도달할 때까지 반복
        action = agent.select_action(state)

        # 다음 상태로 이동 (간단한 예제에서는 상태 및 보상이 간소화되어 있음)
        next_state = min(state + 1, num_states - 1)

        # 보상 및 Q-Table 업데이트
        reward = 0 if next_state != num_states - 1 else 100  # 목표 도달 시 높은 보상
        agent.update_q_table(state, action, reward, next_state)

        total_reward += reward
        state = next_state

    if episode % 100 == 0:
        print(f"에피소드: {episode}, 총 보상: {total_reward}")

# 학습된 Q-Table 출력
print("최종 Q-Table:")
print(agent.q_table)