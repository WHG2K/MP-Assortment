import numpy as np
from src.utils.lp_optimizers import LinearProgramSolver

def test_feasible_case():
    # 可行问题测试
    print("=== 测试可行问题 ===")
    c = np.array([2.0, 1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    
    # 测试PuLP求解器
    pulp_solver = LinearProgramSolver(solver='pulp')
    pulp_val, pulp_sol, pulp_status = pulp_solver.maximize(c, A, b)
    
    print("PuLP结果:")
    print(f"状态: {pulp_status}")
    print(f"最优值: {pulp_val}")
    print(f"最优解: {pulp_sol}")
    
    try:
        # 测试Gurobi求解器
        gurobi_solver = LinearProgramSolver(solver='gurobi')
        gurobi_val, gurobi_sol, gurobi_status = gurobi_solver.maximize(c, A, b)
        
        print("\nGurobi结果:")
        print(f"状态: {gurobi_status}")
        print(f"最优值: {gurobi_val}")
        print(f"最优解: {gurobi_sol}")
        
        print("\n结果比较:")
        print(f"最优值差异: {abs(pulp_val - gurobi_val)}")
        print(f"解向量差异: {np.max(np.abs(pulp_sol - gurobi_sol))}")
        
    except (ImportError, ValueError) as e:
        print("\nGurobi测试跳过:", str(e))

def test_infeasible_case():
    # 不可行问题测试
    print("\n=== 测试不可行问题 ===")
    c = np.array([1.0, 1.0])
    # 矛盾的约束:
    # x1 + x2 >= 2 (转换为 -x1 - x2 <= -2)
    # x1 + x2 <= 1
    # x1, x2 在[0,1]之间
    A = np.array([
        [-1.0, -1.0],  # -x1 - x2 <= -2
        [1.0, 1.0]     # x1 + x2 <= 1
    ])
    b = np.array([-2.0, 1.0])
    
    # 测试PuLP求解器
    pulp_solver = LinearProgramSolver(solver='pulp')
    pulp_val, pulp_sol, pulp_status = pulp_solver.maximize(c, A, b)
    
    print("PuLP结果:")
    print(f"状态: {pulp_status}")
    print(f"最优值: {pulp_val}")
    print(f"最优解: {pulp_sol}")
    
    try:
        # 测试Gurobi求解器
        gurobi_solver = LinearProgramSolver(solver='gurobi')
        gurobi_val, gurobi_sol, gurobi_status = gurobi_solver.maximize(c, A, b)
        
        print("\nGurobi结果:")
        print(f"状态: {gurobi_status}")
        print(f"最优值: {gurobi_val}")
        print(f"最优解: {gurobi_sol}")
        
    except (ImportError, ValueError) as e:
        print("\nGurobi测试跳过:", str(e))

if __name__ == "__main__":
    test_feasible_case()
    test_infeasible_case() 