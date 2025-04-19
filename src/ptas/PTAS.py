import itertools
from typing import List
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class AO_Instance:
    def __init__(self, n, M, lambdas, weights, prices, C):
        self.n = n
        self.M = M
        self.C = C
        self.lambdas = lambdas
        self.weights = weights
        self.prices = prices

    def Get_Opt(self):
        best_rev = 0
        for i in range(self.M, self.n + 1):
            assort_list = self.generate(self.n, i)
            for assort in assort_list:
                current_rev = self.Compute_Rev(assort)
                best_rev = max(best_rev, current_rev)
        return best_rev

    def Get_Opt_Card(self):
        best_rev = 0
        assort_list = self.generate(self.n, self.C)
        for assort in assort_list:
            current_rev = self.Compute_Rev(assort)
            best_rev = max(best_rev, current_rev)
        return best_rev

    def Compute_Rev(self, S):
        total_rev = 0
        for prod in S:
            cp = self.True_Choice_Prob(S, prod)
            total_rev += cp * self.prices[prod]
        return total_rev

    def True_Choice_Prob(self, S, prod):
        w_i = self.weights[prod]
        V_cp = {}
        num_offered = len(S)
        cap = min(num_offered, self.M)

        for m in range(1, cap + 1):
            for j in range(max(num_offered - self.M + m, 1), num_offered + 1):
                all_assorts = self.generate(num_offered, j)
                for assort in all_assorts:
                    state = tuple([m] + list(assort))
                    total_weight = 1 + sum(self.weights[S[i]] for i in assort)

                    if m == 1:
                        cp = w_i / total_weight
                        V_cp[state] = cp
                    else:
                        next_m = 0
                        for i in assort:
                            other_prod = S[i]
                            if other_prod != prod:
                                new_state = tuple([m - 1] + [k for k in assort if k != i])
                                next_m += (self.weights[other_prod] / total_weight) * V_cp.get(new_state, 0)
                        cp = w_i / total_weight + next_m
                        V_cp[state] = cp

        final_cp = 0
        for m in range(1, self.M + 1):
            init_state = tuple([min(num_offered, m)] + list(range(num_offered)))
            final_cp += self.lambdas[m] * V_cp.get(init_state, 0)

        return final_cp

    def Get_C(self, prod, heavy, delta, eps):
        V = {}
        w_i = self.weights[prod]
        num_heavy = len(heavy)

        for m in range(1, self.M + 1):
            for j in range(num_heavy - self.M + m, num_heavy + 1):
                all_H_poss = self.generate(num_heavy, j)
                for H_index in all_H_poss:
                    state = tuple([m] + list(H_index))

                    weight_H = sum(self.weights[heavy[h]] for h in H_index)
                    weight_H_no_i = sum(self.weights[heavy[h]] for h in H_index if heavy[h] != prod)
                    part_1 = 1 / (1 + delta + weight_H)

                    if m == 1:
                        value = part_1 * w_i
                        V[state] = value
                    else:
                        state_part_2 = tuple([m - 1] + list(H_index))
                        part_2 = w_i + max(0, delta - eps * weight_H_no_i) * V.get(state_part_2, 0)

                        part_3 = 0
                        for h_index in H_index:
                            heavy_prod = heavy[h_index]
                            if heavy_prod != prod:
                                state_part_3 = tuple([m - 1] + [k for k in H_index if k != h_index])
                                part_3 += self.weights[heavy_prod] * V.get(state_part_3, 0)

                        value = part_1 * (part_2 + part_3)
                        V[state] = value

        choice_prob = 0
        for m in range(1, self.M + 1):
            init_state = tuple([m] + list(range(num_heavy)))
            choice_prob += self.lambdas[m] * V.get(init_state, 0)

        return choice_prob

    def Get_Delta_Guesses(self, eps, light):
        delta_guesses = [0.0]
        min_weight = min(self.weights[l] for l in light) if len(light) > 0 else 1000
        sum_light = sum(self.weights[l] for l in light)

        current = min_weight
        delta_guesses.append(current)
        while current < sum_light:
            current *= (1 + eps)
            delta_guesses.append(current)

        return delta_guesses

    def Get_H(self, eps):
        num_heavy = np.floor(self.M / eps).astype(int)
        return self.generate(self.n, num_heavy)

    def Get_Light(self, heavy):
        min_weight_H = min(self.weights[h] for h in heavy)
        light_prods_list = [i for i in range(self.n) if self.weights[i] < min_weight_H]
        return light_prods_list

    def generate(self, n, r):
        return list(itertools.combinations(range(n), r))






class MP_MNL_PTAS:

    def __init__(self, test_case_):
        self.test_case = test_case_
        self.best_S = [0] * self.test_case.C

    def solve(self, eps):
        S = []
        H_guesses = self.test_case.Get_H(eps)
        best_IP = 0

        for j, heavy in enumerate(H_guesses):
            # print(j)
            heavy = list(heavy)
            light = self.test_case.Get_Light(heavy)
            num_light = len(light)
            num_heavy = len(heavy)
            delta_guesses = self.test_case.Get_Delta_Guesses(eps, light)

            for delta in delta_guesses:
                rewards_list = []
                weight_list = []

                for l_prod in light:
                    reward = self.test_case.prices[l_prod] * self.test_case.Get_C(l_prod, heavy, delta, eps)
                    weight = self.test_case.weights[l_prod]
                    rewards_list.append(reward)
                    weight_list.append(weight)

                opt_light = self.Solve_Knap_Light(num_light, rewards_list, weight_list, delta, eps, num_heavy)

                total_offered = num_heavy + len(opt_light)
                S = [light[l_index] for l_index in opt_light] + heavy

                if len(S) == self.test_case.C:
                    opt_IP = self.test_case.Compute_Rev(S)
                else:
                    opt_IP = 0

                if opt_IP > best_IP and len(S) == self.test_case.C:
                    best_IP = opt_IP
                    self.best_S = S.copy()

        return best_IP

    def Solve_Knap_Light(self, num_light, rewards, weights, cap, eps, num_heavy):
        opt = []
        num_to_offer = self.test_case.C - num_heavy

        try:
            model = gp.Model("Knap")
            model.setParam('OutputFlag', 0)

            x_list = [model.addVar(vtype=GRB.BINARY, name=f"x_{l}") for l in range(num_light)]

            model.update()

            weight_limit_lower = gp.LinExpr()
            total_light = gp.LinExpr()
            for l in range(num_light):
                weight_limit_lower.add(x_list[l], weights[l])
                total_light.add(x_list[l], 1.0)

            model.addConstr(weight_limit_lower >= cap / (1 + eps), "lower")
            model.addConstr(weight_limit_lower <= cap, "upper")
            model.addConstr(total_light == num_to_offer, "C_const")

            model.setObjective(gp.quicksum(-rewards[l] * x_list[l] for l in range(num_light)), GRB.MINIMIZE)

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                for l in range(num_light):
                    if x_list[l].X == 1:
                        opt.append(l)

            model.dispose()

        except gp.GurobiError as e:
            print(f"Error code: {e.errno}. {e}")

        return opt
