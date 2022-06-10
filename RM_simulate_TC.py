import matplotlib.pylab as plt
import numpy as np

def total_cost(T, T_prime, k, alpha, days_after_t1, C1):
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


def setup_cost(C1, T):
    return C1/T


def holding_cost(T_prime, T, lambda_demand, p1, p2, alpha, C2, days_after_t1):
    S = lambda_demand * T_prime
    t1 = (alpha*S)/(p1-lambda_demand)
    t3 = t1 + days_after_t1/365

    phi = (T - (alpha * S) / (p1 - lambda_demand) - (t3 - t1) + ((t3 - t1) * (p1 - lambda_demand)) / (
                p2 - lambda_demand) - T_prime) / ((1 / (p2 - lambda_demand)) + (1 / lambda_demand))

    A = (phi**2)/(2*(p2-lambda_demand))
    B = (phi**2)/(2*lambda_demand)
    C = 0.5*(t3 - (alpha*S)/(p1-lambda_demand))**2 * (p1 - lambda_demand)
    D = ((t3-t1)**2 * (p1-lambda_demand)**2)/(2*(p2-lambda_demand))

    return C2/T*(A + B + C - D)


def shortage_cost(C3, T, alpha, p1, lambda_demand, T_prime):
    S = lambda_demand * T_prime
    return C3 / (2 * T) * alpha * ((p1 - (1 - alpha) * lambda_demand) / (lambda_demand * (p1 - lambda_demand))) * (S ** 2)


def penalty_cost(T, T_prime, C4, alpha, lambda_demand):
    S = lambda_demand * T_prime
    return C4/T*(1-alpha)*S


def simulate(k, days_after_t1, C1):
    C2 = 2
    C3 = 3.2
    C4 = 4

    p1 = 9200
    p2 = p1 * k
    lambda_demand = 1100

    alpha = 0.75

    T = np.linspace(0, 1, 10)
    T_prime = np.linspace(0, 1, 10)

    lowest_cost = 1000000000000
    optimal_T = None
    optimal_T_prime = None
    optimal_phi = None

    for T_i in T:
        for T_prime_i in T_prime:
            cost, phi = total_cost(T_i, T_prime_i, k, alpha, days_after_t1, C1)
            if cost < lowest_cost:
                lowest_cost = cost
                optimal_T = T_i
                optimal_T_prime = T_prime_i
                optimal_phi = phi

    print("Optimal T: ", optimal_T)
    print("Optimal T_prime: ", optimal_T_prime)
    print("Minimal Cost: ", lowest_cost)
    print("Setup cost: ", setup_cost(C1, optimal_T))
    print("Inventory holding cost: ", holding_cost(optimal_T_prime, optimal_T, lambda_demand, p1, p2, alpha, C2, days_after_t1))
    print("Shortage cost: ", shortage_cost(C3, optimal_T, alpha, p1, lambda_demand, optimal_T_prime))
    print("Penalty cost: ", penalty_cost(optimal_T, optimal_T_prime, C4, alpha, lambda_demand))

    print("Optimal Q: ", optimal_quantity(optimal_T, optimal_T_prime, lambda_demand=1100, alpha=alpha))
    print("Optimal S: ", 1100*optimal_T_prime)
    print("Maximum Stock Level in Cycle: ", optimal_phi)
    print("Operating cost per unit: ", lowest_cost*optimal_T/optimal_quantity(optimal_T, optimal_T_prime, lambda_demand=1100, alpha=alpha))

    return lowest_cost, setup_cost(C1, optimal_T), holding_cost(optimal_T_prime, optimal_T, lambda_demand, p1, p2, alpha, C2, days_after_t1), shortage_cost(C3, optimal_T, alpha, p1, lambda_demand, optimal_T_prime), penalty_cost(optimal_T, optimal_T_prime, C4, alpha, lambda_demand)


if __name__ == "__main__":
    increase = np.array([1, 1.2, 1.4, 1.7, 2.0])

    cost = []
    inv_cost = []
    setup_c = []
    penalty_c = []
    shortage_c = []

    for k in increase:
        print("Optimal policies with p2 =", 9200*k)
        optimal, setup, inv, short, penal = simulate(k, days_after_t1=90, C1=275)
        print("\n")
        cost.append(optimal)
        setup_c.append(setup)
        inv_cost.append(inv)
        shortage_c.append(short)
        penalty_c.append(penal)

    x_axis = increase * 9200
    plt.plot(x_axis, cost, label="$TC(T^*, T'^*)$")
    plt.plot(x_axis, setup_c, label="Setup Cost")
    plt.plot(x_axis, inv_cost, label="Inventory Holding Cost")
    plt.plot(x_axis, shortage_c, label="Shortage Cost")
    plt.plot(x_axis, penalty_c, label="Penalty Cost")
    plt.xlabel("$p_2$")
    plt.ylabel("$costs$")
    plt.legend()
    plt.show()


    # setup_cost_list = [50, 100, 150, 200, 250, 300, 350, 400]
    # for C1 in setup_cost_list:
    #     print("Optimal policies with C1: ", C1)
    #     simulate(k=2, days_after_t1=1, C1=C1)
    #     print("\n")

    # t3 = np.linspace(0,100, 40)
    # cost = []
    # for days_after_t1 in t3:
    #     print("t3: ", days_after_t1)
    #     optimal, setup, inv, short, penal = simulate(k=2, days_after_t1=days_after_t1, C1=275)
    #     print("\n")
    #     cost.append(optimal)
    # plt.plot(t3, cost)
    # plt.ylabel("$TC(T^*, T'^*)$")
    # plt.xlabel("$t_3$")
    # plt.show()
