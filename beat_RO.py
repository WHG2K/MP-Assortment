import numpy as np
import matplotlib.pyplot as plt


class LOP_2p:

    def __init__(self, u, r, b, p):
        self.u = np.array(u).reshape(-1)
        self.r = np.array(r).reshape(-1)
        self.b = b
        self.p = p

    def probs_U_exceed_w(self, w):
        return 1 - np.exp(-np.exp(-(w-self.u + np.euler_gamma)))

    def probs_U_exceed_w_and_U0(self, w):
        """
        Compute the probability that U=u+eps exceeds w and U0
        """
        u = self.u
        w_ = w + np.euler_gamma
        return 1/(1 + np.exp(-u)) * (1 - np.exp(-(np.exp(u)+1)*np.exp(-w_)))


    def w_x(self, x):
        assert len(x) == 2
        p = self.p

        x = np.array(x).reshape(-1)
        w_low = - max(np.abs(self.u)) - 50
        w_high = max(np.abs(self.u)) + 50
        # print(w_low, w_high)
        i = 0
        while (i < 25):
            w = (w_low + w_high) / 2
            probs_matrix = self.probs_U_exceed_w(w)
            viol_cons = probs_matrix @ x - self.b
            # print("w=", w, "viol_cons=", viol_cons)
            if viol_cons < 0:
                w_high = w
            else:
                w_low = w
            # w_high[viol_cons < 0] = w[viol_cons < 0]
            # w_low[viol_cons > 0] = w[viol_cons > 0]
            i += 1

        return (w_low + w_high) / 2
    

    def revenue(self, x):
        assert len(x) == 2
        x = np.array(x).reshape(-1)
        w = self.w_x(x)
        return self.r * self.probs_U_exceed_w_and_U0(w) @ x
    
def transform(t, p):
    if t <= p:
        return [t, 0]
    else:
        return [p, t - p]
    
def w(t, p, b):
    x = transform(t, p)
    return lop.w_x(x)


if __name__ == "__main__":

    p = 0.03
    mu = 2
    u = [-mu, -mu]
    r = [100, 40]
    b = p
    lop = LOP_2p(u, r, b, p)
    # print(lop.probs_U_exceed_w(0))
    # print(lop.probs_U_exceed_w_and_U0(0))

    def f(t):
        return lop.revenue(transform(t, p))

    # t = 0.05
    # x = transform(t, p)
    # print("x=", x)
    # w = lop.w_x(x)
    # print("w=", w)

    t = np.linspace(0, 1, 10000)
    # t = np.linspace(0, 10*p, 10000)
    # t = np.linspace(p, 2*p, 1000)
    y = [f(t_i) for t_i in t]


    # 画折线图
    plt.plot(t, y)  # marker='o' 是为了显示折点
    plt.axhline(y=lop.revenue(transform(1, p)), color='r', linestyle='--', label='y = 0.5')  # 红色虚线
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Plot of rev(t) on [0, 1]')
    plt.grid(True)
    plt.show()


    # # t = np.linspace(p, p+0.1, 1000)
    # w = np.linspace(-10, 10, 1000)
    # y = [g(w_i, mu) for w_i in w]

    # # 画折线图
    # plt.plot(w, y, marker='o')  # marker='o' 是为了显示折点
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # # plt.title('Plot of rev(t) on [0, 1]')
    # plt.grid(True)
    # plt.show()
