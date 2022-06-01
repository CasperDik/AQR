from gurobipy import Model, GRB, quicksum
import matplotlib.pylab as plt
import numpy as np

def total_cost(T, T_prime, k, alpha, days_after_t1):
    C1 = 275
    C2 = 2
    C3 = 3.2
    C4 = 4

    p1 = 9200
    p2 = p1 * k
    lambda_demand = 1100

    S = lambda_demand * T_prime
    t1 = (alpha*S)/(p1-lambda_demand)
    t3 = t1 + days_after_t1/365

    phi = (T - (alpha*S)/(p1-lambda_demand) - (t3 - t1) + ((t3-t1)*(p1-lambda_demand))/(p2-lambda_demand) - T_prime) / ((1/(p2-lambda_demand))+(1/lambda_demand))

    A = (phi**2)/(2*(p2-lambda_demand))
    B = (phi**2)/(2*lambda_demand)
    C = 0.5*(t3 - (alpha*S)/(p1-lambda_demand))**2 * (p1 - lambda_demand)
    D = ((t3-t1)**2 * (p1-lambda_demand)**2)/(2*(p2-lambda_demand))

    total_cost = C1/T + C2/T*(A + B + C - D) + C3/(2*T) * alpha * ((p1 - (1 - alpha)*lambda_demand)/(lambda_demand*(p1 - lambda_demand)))*(S**2) + C4/T*(1-alpha)*S

    return total_cost, phi


def optimal_quantity(T, T_prime, lambda_demand, alpha):
    return lambda_demand*(T - (1-alpha)*T_prime)

def optimize():
    n = np.linspace(0, 1, 1000)

    m = Model("inventory model")

    T = m.addVars(n, vtype=GRB.CONTINUOUS, name="T", lb=0)
    T_prime = m.addVars(n, vtype=GRB.CONTINUOUS, name="T_prime", lb=0)

    C1 = 275
    C2 = 2
    C3 = 3.2
    C4 = 4

    p1 = 9200
    p2 = 1.0*p1 #todo: as input function

    alpha = 0.75
    lambda_demand = 1100

    t3 = 25/365  #todo: as input function

    # todo: fix objective function
    m.setObjective(quicksum(C1/T[i] + C2/T[i]*(lambda_demand**2/p2 * (T[i] - T_prime[j]*(alpha - 1)) + lambda_demand*(T[i] + T_prime[j]
            + t3*(p1/p2 - 1))**2/2*(p2-lambda_demand) + lambda_demand**2/p2 * (T[i] - T_prime[j]*(alpha - 1)) +
            lambda_demand*(T[i] + T_prime[j] + t3*(p1/p2 - 1))**2/(2*lambda_demand) + 0.5*(t3 -
            (alpha*lambda_demand * T_prime[j])/(p1-lambda_demand))**2 * (p1 - lambda_demand) - ((t3-(alpha *
            lambda_demand * T_prime[j])/(p1-lambda_demand))**2 * (p1-lambda_demand)**2)/(2*(p2-lambda_demand))) +
            C3/(2*T[i]) * alpha * ((p1 - (1- alpha)*lambda_demand)/(lambda_demand*(p1-lambda_demand)))*(lambda_demand *
            T_prime[j])**2 + C4/T[i]*(1-alpha)*lambda_demand * T_prime[j] for j in range(len(n)) for i in range(len(n))), GRB.MAXIMIZE)

    m.write("model.lp")
    m.optimize()

def simulate(k):
    alpha = 0.75
    days_after_t1 = 90

    T = np.linspace(0, 1, 2000)
    T_prime = np.linspace(0, 1, 2000)

    lowest_cost = 1000000000
    optimal_T = None
    optimal_T_prime = None
    optimal_phi = None

    for T_i in T:
        for T_prime_i in T_prime:
            cost, phi = total_cost(T_i, T_prime_i, k, alpha, days_after_t1)
            if cost < lowest_cost:
                lowest_cost = cost
                optimal_T = T_i
                optimal_T_prime = T_prime_i
                optimal_phi = phi

    print("Optimal T: ", optimal_T)
    print("Optimal T_prime: ", optimal_T_prime)
    print("Minimal Cost: ", lowest_cost)
    print("Optimal Q: ", optimal_quantity(optimal_T, optimal_T_prime, lambda_demand=1100, alpha=alpha))
    print("Optimal S: ", 1100*optimal_T_prime)
    print("Maximum Stock Level in Cycle: ", optimal_phi)



if __name__ == "__main__":
    increase = [1, 1.2, 1.4, 1.7, 2.0]
    for k in increase:
        print("Optimal policies with p2 =", 9200*k)
        simulate(k)
        print("\n")

