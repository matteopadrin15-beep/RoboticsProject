from sympy import sin, cos, Matrix, sqrt, lambdify,eye
from sympy.abc import alpha, beta,gamma
import matplotlib.pyplot as plt
import math
import numpy as np

J = Matrix([ cos(alpha) + cos(alpha+beta) + cos(alpha + beta + gamma) , sin(alpha) + sin(alpha+beta) + sin(alpha + beta + gamma), alpha + beta + gamma])
Y = Matrix([alpha, beta, gamma])
Jac = J.jacobian(Y) 
Jinv = Jac**-1 # Sì, funziona. Davvero

jinv_func =      lambdify((alpha, beta, gamma), Jinv, 'numpy')
jac_direct_func= lambdify((alpha, beta, gamma), Jac, 'numpy')

#LAMBDA = 0.1
#JPseInv = Jac.T @ (Jac @ Jac.T + LAMBDA * eye(3))**-1 
#j_pseudo_inv_func =      lambdify((alpha, beta, gamma), JPseInv, 'numpy') 
 


TARGET = [1,2,math.pi/2] 
INITIAL = [math.pi/2,math.pi/2,math.pi/2]
ALPHA = 0
EPSILON = [10e-5,10e-5,10e-5]
MAX_ITER = 10000
TOL = 1e-5
SCALAR_EPSILON = 1e-5


def kappa(q1=None,q2=None,q3=None, q=None):
    if q is not None:
        q1, q2, q3 = q
    if None in (q1, q2, q3):
        raise ValueError("Parametri forniti non corretti")
    
    x = np.cos(q1) + np.cos(q1 + q2) + np.cos(q1 + q2 + q3)
    y = np.sin(q1) + np.sin(q1 + q2) + np.sin(q1 + q2 + q3)
    phi = q1 + q2 + q3
    return np.array([x, y, phi])


def newton(alpha=ALPHA):
    q = np.array(INITIAL, dtype=float) 
    target_pos = np.array(TARGET)
    history=[]
    for i in range(MAX_ITER):
        current_pos = kappa(q=q)
        error_vector = target_pos - current_pos    
        if np.linalg.norm(error_vector) < TOL:
            history.append(current_pos[:2])
            return history 
        history.append(current_pos[:2])   
        with np.errstate(divide='ignore', invalid='ignore'): #Per sopprimere il warning 
            inv_j_num = jinv_func(q[0], q[1], q[2])
        q = q + alpha * (inv_j_num @ error_vector)   
    return history

def generalizedNewton(alpha_step=0.1, lam=0.1):
    q = np.array(INITIAL, dtype=float)
    target_pos = np.array(TARGET)
    history = []
    i=0
    J_d=0
    for i in range(MAX_ITER):
        current_pos = kappa(q=q)
        error_vector = target_pos - current_pos
        if np.linalg.norm(error_vector) < TOL:
            history.append(current_pos[:2].copy())
            return history,i
        history.append(current_pos[:2].copy())
        J_n = jac_direct_func(q[0], q[1], q[2])         
        JJT = J_n @ J_n.T                       
        J_d = J_n.T @ np.linalg.inv(JJT + lam * np.eye(3)) 
        q = q + alpha_step * (J_d @ error_vector)
    return history,i

def redundantNewton(alpha_step=0.1, k0=0.5):
    w_sym = sqrt((Jac.T * Jac).det()+SCALAR_EPSILON) # w = root( det JJ )
    grad_w_sym = w_sym.diff(Y)
    w_func = lambdify((alpha, beta, gamma), w_sym, 'numpy') 
    grad_w_func = lambdify((alpha, beta, gamma), grad_w_sym, 'numpy')
    jac_func = lambdify((alpha, beta, gamma), Jac, 'numpy')
    q = np.array(INITIAL, dtype=float) 
    target_pos = np.array(TARGET)
    history = []
    i=0
    for i in range(MAX_ITER):
        current_pos = kappa(q=q)
        history.append(current_pos.copy())
        error_vector = target_pos - current_pos    
        if np.linalg.norm(error_vector) < TOL:
            print(f"Target raggiunto {i} ")
            return history,i 
        J_n = jac_func(q[0], q[1], q[2])
        g_w_n = grad_w_func(q[0], q[1], q[2]) 
        J_pinv = np.linalg.pinv(J_n, rcond=1e-3)
        I = np.eye(len(q))
        P = I - J_pinv @ J_n
        q_dot_task = J_pinv @ error_vector
        q_dot_null = P @ (k0 * g_w_n.T).flatten()
        q = q + alpha_step * (q_dot_task + q_dot_null)      
    return history,i


def gradient(alpha=ALPHA):
    q = np.array(INITIAL, dtype=float) 
    target_pos = np.array(TARGET)
    history=[]
    for i in range(MAX_ITER):
        current_pos = kappa(q=q)
        error_vector = target_pos - current_pos
        if np.linalg.norm(error_vector) < TOL:
            history.append(current_pos[:2])
            return history    
        history.append(current_pos[:2])
        j_num = jac_direct_func(q[0], q[1], q[2])
        q = q + alpha * (j_num.T @ error_vector)
    return history

def plot(history,lab): 
    target_pos = np.array(TARGET)
    history = np.array(history)
    plt.figure(figsize=(8, 6))
    plt.plot(history[:, 0], history[:, 1], '-o', color='orange', markersize=3, label='Path', alpha=0.6)
    init_str = ", ".join([f"{x:.2f}" for x in INITIAL])
    plt.plot(history[0, 0], history[0, 1], 'ro', label=f'Initial: ({init_str})')
    plt.plot(target_pos[0], target_pos[1], 'gs', label='Target')
    plt.title(f'End effector trajectory ({lab})' )
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal')
    plt.show() 
    return plt

def merge_plots(p1, p2, p3, p4, labels, alphas, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)
    data = [p1, p2, p3, p4]
    
    for i, ax in enumerate(axs.flat):
        h = np.array(data[i]) 
        if h.ndim < 2:
            print(f"Errore: nel set di dati")
            continue
        target_pos = np.array(TARGET)
        ax.plot(h[:, 0], h[:, 1], '-o', color='orange', markersize=3, label='Path', alpha=0.6)
        ax.plot(h[0, 0], h[0, 1], 'ro', label='Initial')
        ax.plot(target_pos[0], target_pos[1], 'gs', label='Target')

        ax.set_title(f"{labels[i]} (alpha={alphas[i]})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.axis('equal')

    plt.tight_layout()
    plt.show()

def compare_plots(cases, title):
    """
    cases: lista di 4 tuple (h_gen, h_red, label, alpha)
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    for ax, (h_gen, h_red, label, a) in zip(axs.flat, cases):
        h_gen = np.array(h_gen)
        h_red = np.array(h_red)[:, :2]

        ax.plot(h_gen[:, 0], h_gen[:, 1], '-o', color='steelblue',
                markersize=2, alpha=0.7, label='Generalized')
        ax.plot(h_red[:, 0], h_red[:, 1], '-o', color='tomato',
                markersize=2, alpha=0.7, label='Redundant + null space')
        ax.plot(h_gen[0, 0], h_gen[0, 1], 'ko', markersize=6, label='Start')
        ax.plot(TARGET[0], TARGET[1], 'gs', markersize=8, label='Target')

        ax.set_title(f"{label} (alpha={a})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.axis('equal')

    plt.tight_layout()
    plt.show()

def main():
    global ALPHA
    ALPHA = 0.5
    hnew1pi = newton(ALPHA)
    hgrad1pi= gradient(ALPHA)
    print("="*20)
    print(f"Fine di newton alpha = 0.5 {hnew1pi[-10:]}")
    print("="*20)
    print(len(hnew1pi))
    #plot(hnew1pi,"QUELLO VERO")
    alpha = 0.1
    hnew2pi=newton(alpha)
    hgrad2pi = gradient(alpha)
 

    labels = ["Newton", "Gradient", "Newton", "Gradient"]
    alphas = [0.5, 0.5, 0.1, 0.1]
    title = "Plots with starting point [pi/2, pi/2, pi/2]^T"
    merge_plots(hnew1pi,hgrad1pi,hnew2pi,hgrad2pi,labels,alphas,title)
    # =====
    global INITIAL
    INITIAL = [0,0,0]
    alpha = 0.5
    hnew10 = newton(alpha)
    hgrad10= gradient(alpha)

    alpha = 0.1
    hnew20  = newton(alpha)
    hgrad20 = gradient(alpha)
 

    labels = ["Newton", "Gradient", "Newton", "Gradient"]
    alphas = [0.5, 0.5, 0.1, 0.1] 
    title = "Plots with starting point [0,0,0]^T"
    merge_plots(hnew10,hgrad10,hnew20,hgrad20,labels,alphas,title)

    # =====
    alpha = 0.5
    hnew10reviwed,_=  redundantNewton(alpha)
    alpha = 0.1
    hnew20reviwed,_= redundantNewton(alpha)
    title = "Plots with starting point [0,0,0]^T (redundant)"
    merge_plots(hnew10reviwed,hgrad10,hnew20reviwed,hgrad20,labels,alphas,title)

    INITIAL = [math.pi/2, math.pi/2, math.pi/2]
    cases_pi = [
        (generalizedNewton(0.5)[0], redundantNewton(0.5)[0], "start=pi/2", 0.5),
        (generalizedNewton(0.1)[0], redundantNewton(0.1)[0], "start=pi/2", 0.1),
    ]

    INITIAL = [0, 0, 0]
    cases_0 = [
        (generalizedNewton(0.5)[0], redundantNewton(0.5)[0], "start=0", 0.5),
        (generalizedNewton(0.1)[0], redundantNewton(0.1)[0], "start=0", 0.1),
    ]

    compare_plots(cases_pi + cases_0, "Generalized vs Redundant Newton")


    #Comparison
    for start, label in [([math.pi/2]*3, "pi/2"), ([0,0,0], "0")]:
        INITIAL = start
        for alpha in [0.5, 0.1]:
            h_gen, i_gen = generalizedNewton(alpha)
            h_red, i_red = redundantNewton(alpha)
            print(f"Start={label}, alpha={alpha} : Generalized: {i_gen} iter | Redundant: {i_red} iter")
    print("\n", "="*10)
    print(f"{Jac=}")
    print("\n", "="*10)
    print(f"Il determiannte è {Jac.det()}")


    

if __name__=="__main__":
    main()





