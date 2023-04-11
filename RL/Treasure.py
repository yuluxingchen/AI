import numpy as np
from gym.envs.toy_text import discrete

# 定义动作空间
up = 0
right = 1
down = 2
left = 3
# 定义宝藏区
done_location = 8


# 定义网格世界环境模型
class GridworldEnv(discrete.DiscreteEnv):
    def render(self, mode="human"):
        pass

    def __init__(self, shape=None):
        if shape is None:
            shape = [5, 5]
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list or tuple of length 2')
        self.shape = shape
        # 状态个数：行 * 列
        self.nS = np.prod(shape)
        # 动作个数
        self.nA = 4
        MAX_Y = shape[0]
        MAX_X = shape[1]
        P = {}
        # 创建一个5 * 5的表格
        grid = np.arange(self.nS).reshape(shape)
        # 多重索引排序，如（1,3)
        iter = np.nditer(grid, flags=['multi_index'])

        while not iter.finished:
            s = iter.iterindex
            y, x = iter.multi_index
            P[s] = {a: [] for a in range(self.nA)}
            is_done = lambda s: s == done_location
            if is_done(s):
                reward = 0.0
                for i in range(self.nA):
                    P[s][i] = [(1, s, reward, True)]
            else:
                reward = -1.0
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                P[s][up] = [(1, ns_up, reward, is_done(ns_up))]
                P[s][right] = [(1, ns_right, reward, is_done(ns_right))]
                P[s][down] = [(1, ns_down, reward, is_done(ns_down))]
                P[s][left] = [(1, ns_left, reward, is_done(ns_left))]
            iter.iternext()
        isd = np.ones(self.nS) / self.nS
        self.P = P
        super(GridworldEnv, self).__init__(self.nS, self.nA, P, isd)


# 根据传入的四个行为选择值函数最大的索引，返回的是一个索引数组和一个行为策略
def get_max_index(action_values):
    indexs = []
    policy_arr = np.zeros(len(action_values))
    action_max_value = np.max(action_values)
    for i in range(len(action_values)):
        action_value = action_values[i]
        if action_value == action_max_value:
            indexs.append(i)
            policy_arr[i] = 1
    return indexs, policy_arr


def policy_eval(policy, env, discount_factor=1, threshold=0.00001):
    V = np.zeros(env.nS)
    i = 0
    while True:
        value_delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            value_delta = max(value_delta, np.abs(v - V[s]))
            V[s] = v
        i += 1
        if value_delta < threshold:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1):
    policy = np.ones([env.nS, env.nA]) / env.nA
    global i_num
    global v_num
    i_num = 0
    v_num = 1
    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        policy_stable = True

        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a_arr, policy_arr = get_max_index(action_values)

            if chosen_a not in best_a_arr:
                policy_stable = False
            policy[s] = policy_arr
        i_num = i_num + 1
        if policy_stable:
            print(i_num)
            return policy, V


def value_iteration(env, threshold=0.0001, discount_factor=1):
    global i_num
    i_num = 0
    def one_step_lookahead(state, V):
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                q[a] += prob * (reward + discount_factor * V[next_state])
        return q

    V = np.zeros(env.nS)

    while True:
        delta = 0
        for s in range(env.nS):
            q = one_step_lookahead(s, V)
            best_action_value = np.max(q)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        i_num += 1
        if delta < threshold:
            break
    print(i_num)
    policy = np.zeros([env.nS, env.nA])

    for s in range(env.nS):
        q = one_step_lookahead(s, V)
        best_a_arr, policy_arr = get_max_index(q)
        policy[s] = policy_arr
    return policy, V


def change_policy(policys):
    action_tuple = []
    for policy in policys:
        indexs, policy_arr = get_max_index(policy)
        action_tuple.append(tuple(indexs))
    return action_tuple


if __name__ == '__main__':
    env = GridworldEnv()
    policy, v = policy_improvement(env)
    policy1, v1 = value_iteration(env)
    update_policy_type = change_policy(policy)
    update_policy_type1 = change_policy(policy1)
    print(policy)
    print(policy1)
    print(v)
    print(v1)
    print(update_policy_type)
    print(update_policy_type1)
