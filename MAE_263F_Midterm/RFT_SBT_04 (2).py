##  Code Compiled by Jonathan Gray, 11/10/24  ##


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

def crossMat(a):
    A = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return A

# Define the head motion function for sinusoidal transverse motion
def apply_head_motion(t, amplitude, frequency):
    omega = 2 * np.pi * frequency
    y_head = amplitude * np.sin(omega * t)
    vy_head = amplitude * omega * np.cos(omega * t)
    return y_head, vy_head

def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the derivative of bending energy E_k^b with respect to
    x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dF : np. 
        Derivative of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]

    # Gradient of bending energy
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / l_k

    return dF

def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the Hessian (second derivative) of bending energy E_k^b
    with respect to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dJ : np.ndarray
        Hessian of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]

    # Compute the Hessian (second derivative of kappa)
    DDkappa1 = np.zeros((6, 6))

    norm2_e = norm_e**2
    norm2_f = norm_f**2

    Id3 = np.eye(3)

    # Helper matrices for second derivatives
    tt_o_tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    kb_o_d2e = np.outer(kb, m2e)

    D2kappa1De2 = (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tf_c_d2t_o_tt.T) / norm2_e - \
                  kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te)) + \
                  (kb_o_d2e + kb_o_d2e.T) / (4 * norm2_e)

    tmp = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = te_c_d2t_o_tt.T
    kb_o_d2f = np.outer(kb, m2f)

    D2kappa1Df2 = (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + te_c_d2t_o_tt.T) / norm2_f - \
                  kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf)) + \
                  (kb_o_d2f + kb_o_d2f.T) / (4 * norm2_f)
    D2kappa1DeDf = -kappa1 / (chi * norm_e * norm_f) * (Id3 + np.outer(te, tf)) \
                  + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + \
                  tt_o_te_c_d2t - crossMat(tilde_d2))
    D2kappa1DfDe = D2kappa1DeDf.T

    # Populate the Hessian of kappa
    DDkappa1[0:2, 0:2] = D2kappa1De2[0:2, 0:2]
    DDkappa1[0:2, 2:4] = -D2kappa1De2[0:2, 0:2] + D2kappa1DeDf[0:2, 0:2]
    DDkappa1[0:2, 4:6] = -D2kappa1DeDf[0:2, 0:2]
    DDkappa1[2:4, 0:2] = -D2kappa1De2[0:2, 0:2] + D2kappa1DfDe[0:2, 0:2]
    DDkappa1[2:4, 2:4] = D2kappa1De2[0:2, 0:2] - D2kappa1DeDf[0:2, 0:2] - \
                         D2kappa1DfDe[0:2, 0:2] + D2kappa1Df2[0:2, 0:2]
    DDkappa1[2:4, 4:6] = D2kappa1DeDf[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 0:2] = -D2kappa1DfDe[0:2, 0:2]
    DDkappa1[4:6, 2:4] = D2kappa1DfDe[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 4:6] = D2kappa1Df2[0:2, 0:2]

    # Hessian of bending energy
    dkappa = kappa1 - kappaBar
    dJ = 1.0 / l_k * EI * np.outer(gradKappa, gradKappa)
    dJ += 1.0 / l_k * dkappa * EI * DDkappa1

    return dJ

def getFb(q, EI, deltaL):
    """
    Compute the bending force and Jacobian of the bending force.

    Parameters:
    q : np.ndarray
        A vector of size 6 containing the coordinates [x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, y_{k+1}].
    EI : float
        The bending stiffness.
    deltaL : float
        The Voronoi length.

    Returns:
    Fb : np.ndarray
        Bending force (vector of size 6).
    Jb : np.ndarray
        Jacobian of the bending force (6x6 matrix).
    """

    ndof = q.size # number of DOF
    nv = int(ndof / 2) # number of nodes

    # Initialize bending force as a zero vector of size 6
    Fb = np.zeros(ndof)

    # Initialize Jacobian of bending force as a 6x6 zero matrix
    Jb = np.zeros((ndof, ndof))

    for k in range(1,nv-1): # loop over all nodes except the first and last
        # Extract coordinates from q
        xkm1 = q[2*k-2]
        ykm1 = q[2*k-1]
        xk = q[2*k]
        yk = q[2*k+1]
        xkp1 = q[2*k+2]
        ykp1 = q[2*k+3]
        ind = np.arange(2*k-2,2*k+4)

        # Compute the gradient of bending energy
        gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)

        # Update bending force
        Fb[ind] = Fb[ind] - gradEnergy

        # Compute the Hessian of bending energy
        hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)

        # Update Jacobian matrix
        Jb[np.ix_(ind, ind)] = Jb[np.ix_(ind, ind)] - hessEnergy

    return Fb, Jb

def gradEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    Calculate the gradient of the stretching energy with respect to the coordinates.

    Args:
    - xk (float): x coordinate of the current point
    - yk (float): y coordinate of the current point
    - xkp1 (float): x coordinate of the next point
    - ykp1 (float): y coordinate of the next point
    - l_k (float): reference length
    - EA (float): elastic modulus

    Returns:
    - F (np.array): Gradient array
    """
    F = np.zeros(4)
    F[0] = -(1.0 - np.sqrt((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0) / l_k) * ((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0)**(-0.5) / l_k * (-2.0 * xkp1 + 2.0 * xk)
    F[1] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (-0.2e1 * ykp1 + 0.2e1 * yk)
    F[2] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * xkp1 - 0.2e1 * xk)
    F[3] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * ykp1 - 0.2e1 * yk)

    F = 0.5 * EA * l_k * F  # Scale by EA and l_k

    return F

def hessEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    This function returns the 4x4 Hessian of the stretching energy E_k^s with
    respect to x_k, y_k, x_{k+1}, and y_{k+1}.
    """
    J = np.zeros((4, 4))  # Initialize the Hessian matrix
    J11 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J12 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (-2 * ykp1 + 2 * yk) / 0.2e1
    J13 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * xkp1 - 2 * xk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J14 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J22 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J23 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * xkp1 - 2 * xk) / 0.2e1
    J24 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * ykp1 - 2 * yk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J33 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * xkp1 - 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J34 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (2 * xkp1 - 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J44 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * ykp1 - 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k

    J = np.array([[J11, J12, J13, J14],
                   [J12, J22, J23, J24],
                   [J13, J23, J33, J34],
                   [J14, J24, J34, J44]])

    J *= 0.5 * EA * l_k

    return J

def getFs(q, EA, deltaL):
    ndof = q.size # number of DOF
    nv = int(ndof / 2) # number of nodes

    # Initialize bending force as a zero vector of size 6
    Fs = np.zeros(ndof)

    # Initialize Jacobian of bending force as a 6x6 zero matrix
    Js = np.zeros((ndof, ndof))

    for k in range(0,nv-1): # loop over all nodes except the last
      xkm1 = q[2*k]
      ykm1 = q[2*k+1]
      xk = q[2*k+2]
      yk = q[2*k+3]
      ind = np.arange(2*k,2*k+4)

      # Compute the gradient of stretching energy
      gradEnergy = gradEs(xkm1, ykm1, xk, yk, deltaL, EA)
      Fs[ind] = Fs[ind] - gradEnergy

      # Compute the Hessian of bending energy
      hessEnergy = hessEs(xkm1, ykm1, xk, yk, deltaL, EA)
      Js[np.ix_(ind, ind)] = Js[np.ix_(ind, ind)] - hessEnergy

    return Fs, Js

# Add Resistive Force Theory (RFT) to objfun
# Add hydrodynamic force contribution to objfun
# Add hydrodynamic force contributions based on SBT to objfun
# Add hydrodynamic force contributions based on SBT to objfun
def objfun(q_guess, q_old, u_old, dt, tol, maximum_iter,
           m, mMat,  # inertia
           EI, EA,   # elastic stiffness
           W, C,     # external force
           deltaL,
           free_index): # free_index indicates the DOFs that evolve under equations of motion

    # Set the drag coefficients for SBT
    C_t = 0.2  # Tangential drag coefficient for slender body theory
    C_n = 20  # Normal drag coefficient for slender body theory
    hydrodynamic_influence_range = 3  # Range of segments to consider for hydrodynamic interaction

    q_new = q_guess.copy()
    iter_count = 0
    error = tol * 10
    flag = 1

    # Initialize array to store hydrodynamic force magnitudes
    hydrodynamic_force_magnitudes = np.zeros(nv)
    axial_force_magnitudes = np.zeros(nv)

    while error > tol:
        Fb, Jb = getFb(q_new, EI, deltaL)
        Fs, Js = getFs(q_new, EA, deltaL)

        # Viscous force
        Fv = -C @ (q_new - q_old) / dt
        Jv = -C / dt

        # Initialize hydrodynamic force array
        Fhydro = np.zeros_like(q_new)

        # Calculate SBT-based hydrodynamic forces
        for k in range(1, nv - 1):  # Skip the first and last nodes
            xk, yk = q_new[2*k], q_new[2*k+1]
            vx, vy = (q_new[2*k] - q_old[2*k]) / dt, (q_new[2*k+1] - q_old[2*k+1]) / dt
            velocity_k = np.array([vx, vy])

            # Local tangent and normal
            xkp1, ykp1 = q_new[2*(k+1)], q_new[2*(k+1)+1]
            tangent = np.array([xkp1 - xk, ykp1 - yk])
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            # Tangential and normal velocities
            v_tangential = np.dot(velocity_k, tangent)
            v_normal = np.dot(velocity_k, normal)

            # Local drag forces based on SBT
            Fhydro[2*k] += -C_t * v_tangential * tangent[0] - C_n * v_normal * normal[0]
            Fhydro[2*k+1] += -C_t * v_tangential * tangent[1] - C_n * v_normal * normal[1]

            # Hydrodynamic interaction forces with nearby segments
            for j in range(max(1, k - hydrodynamic_influence_range), min(nv - 1, k + hydrodynamic_influence_range + 1)):
                if j == k:
                    continue  # Skip self-interaction

                xj, yj = q_new[2*j], q_new[2*j+1]
                vj_x, vj_y = (q_new[2*j] - q_old[2*j]) / dt, (q_new[2*j+1] - q_old[2*j+1]) / dt
                velocity_j = np.array([vj_x, vj_y])

                # Relative position and velocity
                r = np.array([xj - xk, yj - yk])
                r_norm = np.linalg.norm(r)
                if r_norm == 0:
                    continue  # Avoid division by zero

                r_hat = r / r_norm
                v_rel = velocity_j - velocity_k

                # Interaction forces (decay with distance to approximate hydrodynamic influence)
                interaction_force = (C_n * np.dot(v_rel, r_hat) * r_hat +
                                     C_t * (v_rel - np.dot(v_rel, r_hat) * r_hat)) / r_norm

                # Apply interaction force to both segment k and j (equal and opposite forces)
                Fhydro[2*k:2*k+2] += interaction_force
                Fhydro[2*j:2*j+2] -= interaction_force


            # Store the magnitude of the hydrodynamic force for color mapping
            hydrodynamic_force_magnitudes[k] = np.linalg.norm(Fhydro[2*k:2*k+2])
            axial_force_magnitudes[k] = np.linalg.norm(Fs[2*k:2*k+2])

        # Equation of motion with hydrodynamic forces
        f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fb + Fs + W + Fv + Fhydro)

        # Jacobian manipulation
        J = mMat / dt**2 - (Jb + Js + Jv)

        f_free = f[free_index]
        J_free = J[np.ix_(free_index, free_index)]

        dq_free = np.linalg.solve(J_free, f_free)
        q_new[free_index] = q_new[free_index] - dq_free

        error = np.linalg.norm(f_free)
        iter_count += 1
        print(f'Iter={iter_count-1}, error={error:.6e}')

        if iter_count > maximum_iter:
            flag = -1
            return q_new, flag, hydrodynamic_force_magnitudes, axial_force_magnitudes

    return q_new, flag, hydrodynamic_force_magnitudes, axial_force_magnitudes

# Inputs and parameters
nv = 16  # Number of vertices
ndof = 2 * nv
dt = 1e-2
RodLength = 2.2
deltaL = RodLength / (nv - 1)
all_DOFs = np.arange(ndof)
free_index = all_DOFs

# Set head motion parameters
head_amplitude = .3
head_frequency = 2

# Radius of nodes, material properties, etc.
free_index = all_DOFs
Head_x = 0.0
Head_index = np.array([0, int(np.round(Head_x / deltaL))])

# Radius of spheres
R = np.zeros(nv)  # Vector of size N - Radius of N nodes
R[:] = 0.05
midNode = int((nv + 1) / 2)

for g in range(nv):
    R[g] = R[g] - (((g + 1) / nv) ** 9) * 0.05

# Densities
rho_metal = 2700
rho_gl = 0.0
rho = rho_metal - rho_gl

# Cross-sectional radius of rod
ro = 1.3e-2
ri = 1.1e-2

# Young's modulus
Y = 7e10

# Viscosity
visc = .75

# Maximum number of iterations in Newton Solver
maximum_iter = 100

# Total simulation time (it exits after t=totalTime)
totalTime = 5

# Utility quantities
ne = nv - 1
EI = Y * np.pi * (ro ** 4 - ri ** 4) / 12
EA = Y * np.pi * (ro ** 2 - ri ** 2)

# Tolerance on force function
tol = EI / RodLength ** 2 * 1e-3

# Geometry of the rod
nodes = np.zeros((nv, 2))
for c in range(nv):
    nodes[c, 0] = c * RodLength / ne

# Compute Mass
m = np.zeros(ndof)
for k in range(nv):
    m[2 * k] = 4 / 3 * np.pi * R[k] ** 3 * rho_metal  # Mass for x_k
    m[2 * k + 1] = m[2 * k]  # Mass for y_k

mMat = np.diag(m)  # Convert into a diagonal matrix

# Weightless
W = np.zeros(ndof)

# Viscous damping matrix, C
C = np.zeros((ndof, ndof))
for k in range(nv):
    C[2 * k, 2 * k] = 6 * np.pi * visc * R[k]
    C[2 * k + 1, 2 * k + 1] = 6 * np.pi * visc * R[k]

# Initial conditions
q0 = np.zeros(ndof)
for c in range(nv):
    q0[2 * c] = nodes[c, 0]
    q0[2 * c + 1] = nodes[c, 1]

q = q0.copy()
u = (q - q0) / dt

# Number of time steps
Nsteps = round(totalTime / dt)
ctime = 0

# Initialize arrays to store results for plotting
all_pos = np.zeros(Nsteps)
all_v = np.zeros(Nsteps)
midAngle = np.zeros(Nsteps)
head_x_positions = np.zeros(Nsteps)
head_x_velocity = np.zeros(Nsteps)

# Set up the plot for animation
fig, (ax1, ax2) = plt.subplots(2, 1)
cmap = cm.get_cmap('coolwarm')

# Adjust space between subplots
plt.subplots_adjust(hspace=0.6)  # Adjust hspace

line1, = ax1.plot([], [], 'k-', linewidth=1.5, zorder = 1)  # Line connecting scatter points, 'k-' is black line
line2, = ax2.plot([], [], 'k-', linewidth=1.5, zorder = 1)  # Line connecting scatter points, 'k-' is black line

# Initialize scatter plots for both hydrodynamic and axial forces
sc1 = ax1.scatter([], [], cmap=cmap, edgecolors='k',zorder = 2)
sc2 = ax2.scatter([], [], cmap=cmap, edgecolors='k',zorder = 2)

# Configure axis labels and limits
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')


#ax1.set_xlim(-0.5-RodLength*4, RodLength + 0.5)
#ax1.set_ylim(-1, 1)
#ax2.set_xlim(-0.5-RodLength*4, RodLength + 0.5)
#ax2.set_ylim(-1, 1)

# Update function for FuncAnimation
def update_plot(timeStep):
    global q, q0, u, ctime
    print(f't={ctime:.6f}')

    # Apply head motion
    y_head, vy_head = apply_head_motion(ctime, head_amplitude, head_frequency)
    q[1] = y_head
    u[1] = vy_head

    # Set plot limits based on initial data (adjust as needed)
    ax1.set_xlim(q[0]-0.5, q[-2]+0.5)
    ax1.set_ylim(-1, 1)
    
    ax2.set_xlim(q[0]-0.5, q[-2]+0.5)
    ax2.set_ylim(-1, 1)


    # Solve for new configuration with updated head motion
    q, error, hydrodynamic_force_magnitudes, axial_force_magnitudes = objfun(
        q, q0, u, dt, tol, maximum_iter, m, mMat, EI, EA, W, C, deltaL, free_index
    )

    if error < 0:
        print('Could not converge. Stopping animation.')
        ani.event_source.stop()
        return

    # Update velocities and positions for the next step
    u = (q - q0) / dt
    ctime += dt
    q0 = q

    # Store the head node's x-position
    head_x_positions[timeStep] = q[0]
    head_x_velocity[timeStep] = u[0]

    # Normalize force magnitudes for color mapping
    normalized_hydro_magnitudes = hydrodynamic_force_magnitudes / np.max(hydrodynamic_force_magnitudes)
    normalized_axial_magnitudes = axial_force_magnitudes / np.max(axial_force_magnitudes)

    # Extract x and y positions
    x1 = q[::2]
    x2 = q[1::2]

    # Update line to connect scatter points
    line1.set_data(x1, x2)
    line2.set_data(x1, x2)

    # Update scatter plots with new data
    sc1.set_offsets(np.c_[x1, x2])
    sc1.set_array(normalized_hydro_magnitudes)
    sc2.set_offsets(np.c_[x1, x2])
    sc2.set_array(normalized_axial_magnitudes)


    # Update titles with current time
    ax1.set_title(f'Hydrodynamic Force Magnitude at t={ctime:.6f}')
    ax2.set_title(f'Axial Force Magnitude at t={ctime:.6f}')

    return sc1, sc2, line1, line2

# Run animation
ani = FuncAnimation(fig, update_plot, frames=Nsteps, blit=False, repeat=False)
plt.show()
# Save the animation
ani.save(f'{nv}nodes_{dt}inc_{head_amplitude}HA_{head_frequency}HF_{3}HIR.gif', writer='ffmpeg', fps=200)

plt.close()

# Plot results

t = np.linspace(0, totalTime, Nsteps)

# New plot for head x-position vs. time
plt.figure(5)
plt.plot(t, head_x_positions, 'b')
plt.xlabel('Time, t [s]')
plt.ylabel('Head X Position, x [m]')
plt.title('X-Position of Head Node Over Time')
plt.savefig('head_x_position.png')

# New plot for head x-velocity vs. time
plt.figure(6)
plt.plot(t, head_x_velocity, 'b')
plt.xlabel('Time, t [s]')
plt.ylabel('Head X Velocity, x [m]')
plt.title('X-Velocity of Head Node Over Time')
plt.savefig('head_x_position.png')

reynolds_numbers = (rho * np.abs(head_x_velocity) * RodLength) / visc
# Create a plot for Reynolds number over time
plt.figure()
plt.plot(t, reynolds_numbers, 'g')
plt.xlabel('Time, t [s]')
plt.ylabel('Reynolds Number, Re')
plt.title('Reynolds Number Over Time')
plt.grid(True)
plt.show()






