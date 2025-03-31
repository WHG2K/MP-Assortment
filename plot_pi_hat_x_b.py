import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.models import MPAssortSurrogate
from src.utils.distributions import GumBel
from tqdm import tqdm

# Set LaTeX-friendly plot parameters
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 12,
    'figure.figsize': (6, 4),  # Standard LaTeX width
    'figure.dpi': 300
})

def plot_pi_hat_x_b():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Problem parameters
    N = 15
    C = (8, 8)
    
    # Generate random instance
    u = np.random.normal(0, 1, N)  # utilities
    w = np.exp(u)
    w_max = np.max(w)
    r = w_max - w  # revenues
    
    # random assortment
    x = np.zeros(N)
    x[:C[0]] = 1
    np.random.shuffle(x)
    
    # Generate b values
    b_values = np.linspace(0, 8, 81)  # adjust range as needed
    pi_hat_values = []
    
    # Calculate pi_hat(x(b)) for each b
    distr = GumBel()
    for b in tqdm(b_values):
        sp = MPAssortSurrogate(u, r, b, distr, C)
        pi_hat_values.append(sp._pi_hat(x))
    
    # Create figure with LaTeX-friendly size
    plt.figure()
    plt.plot(b_values, pi_hat_values, 'b-', linewidth=1.5)
    
    # LaTeX-style labels
    plt.xlabel('$b$')
    plt.ylabel('$\hat{\pi}(x;b)$')
    plt.title('$\hat{\pi}(x;b)$ vs $b$')
    
    # Clean layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig('pi_hat_x_b.pdf', format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_pi_hat_x_b() 