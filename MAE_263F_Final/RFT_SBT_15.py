##  Code Compiled by Jonathan Gray, 11/10/24  ##
##  Code Updated by Jonathan Gray 11/25/24    ##

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import pandas as pd

def crossMat(a):
    A = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return A


def apply_head_motion(t, amplitude, frequency):
    omega = 2 * np.pi * frequency # omega = 2pi
    y_head = amplitude * np.sin(omega * t) # y = A*sin(omega*t)
    vy_head = amplitude * omega * np.cos(omega * t) # vy = amplitude*omega*cos(omega*t)
    return y_head, vy_head


def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI): # Legacy Code from Khalid
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

def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI): # Legacy Code from Khalid
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

def getFb(q, EI, deltaL, R): # Legacy Code from Khalid
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
        gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI*((R[k]/R[0])**4))

        # Update bending force
        Fb[ind] = Fb[ind] - gradEnergy

        # Compute the Hessian of bending energy
        hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI*((R[k]/R[0])**4))

        # Update Jacobian matrix
        Jb[np.ix_(ind, ind)] = Jb[np.ix_(ind, ind)] - hessEnergy

    return Fb, Jb

def gradEs(xk, yk, xkp1, ykp1, l_k, EA): # Legacy Code from Khalid
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

def hessEs(xk, yk, xkp1, ykp1, l_k, EA): # Legacy code from Khalid
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

def getFs(q, EA, deltaL,R): # Legacy code from Khalid
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
      gradEnergy = gradEs(xkm1, ykm1, xk, yk, deltaL, EA*((R[k]/R[0])**2))
      Fs[ind] = Fs[ind] - gradEnergy

      # Compute the Hessian of bending energy
      hessEnergy = hessEs(xkm1, ykm1, xk, yk, deltaL, EA*((R[k]/R[0])**2))
      Js[np.ix_(ind, ind)] = Js[np.ix_(ind, ind)] - hessEnergy

    return Fs, Js

def getRFT(q, q_old, dt, R, nv, rho_gl): # Get hydrodynamic forces from RFT
    """
    Compute hydrodynamic forces and their Jacobian for all nodes.

    Parameters:
    q : np.ndarray
        Current positions of nodes (size 2 * nv).
    q_old : np.ndarray
        Previous positions of nodes (size 2 * nv).
    dt : float
        Time step.
    C_t : float
        Tangential drag coefficient.
    C_n : float
        Normal drag coefficient.
    hydrodynamic_influence_range : int
        Number of neighboring segments to consider for hydrodynamic interaction.
    nv : int
        Number of vertices.

    Returns:
    Fhydro : np.ndarray
        Hydrodynamic force array (size 2 * nv).
    Jhydro : np.ndarray
        Jacobian of hydrodynamic forces (2 * nv x 2 * nv).
    """
    Fhydro = np.zeros_like(q)
    Ftx = 0
    Fnx = 0
    Fx = 0
    Jhydro = np.zeros((len(q), len(q)))

    for k in range(nv):  # Loop over nodes, skipping first
        xk, yk = q[2 * k], q[2 * k + 1] # 2 DOF per node
        vx, vy = (q[2 * k] - q_old[2 * k]) / dt, (q[2 * k + 1] - q_old[2 * k + 1]) / dt # Compute velocity
        velocity_k = np.array([vx, vy])
        C_n = 4 * np.pi * mu / (np.log(RodLength / R[k]) + 1/2)
        C_t = 2 * np.pi * mu / (np.log(RodLength / R[k]) - 1/2)
        if k == nv - 1:
            xkm1, ykm1 = q[2 * (k - 1)], q[2 * (k - 1) + 1] # Store prev node
            tangent = np.array([xk - xkm1, yk - ykm1]) # Compute tangent vector as direction of segnment
            tangent /= np.linalg.norm(tangent) # Normalize tangent
            normal = np.array([-tangent[1], tangent[0]])
        elif k == 0:
            xkp1, ykp1 = q[2 * (k + 1)], q[2 * (k + 1) + 1]  # Store prev node
            tangent = np.array([xkp1 - xk, ykp1 - yk]) # Compute tangent vector as direction of segnment
            tangent /= np.linalg.norm(tangent) # Normalize tangent
            normal = np.array([-tangent[1], tangent[0]])
        else:
            # Local tangent and normal
            xkp1, ykp1 = q[2 * (k + 1)], q[2 * (k + 1) + 1] # Store next node
            xkm1, ykm1 = q[2 * (k - 1)], q[2 * (k - 1) + 1] # Store next node
            tangentp1 = np.array([xkp1 - xk, ykp1 - yk]) # Compute tangent vector as direction of segnment
            tangentp1 /= np.linalg.norm(tangentp1) # Normalize tangent
            tangentm1 = np.array([xk - xkm1, yk - ykm1]) # Compute tangent vector as direction of segnment
            tangentm1 /= np.linalg.norm(tangentm1) # Normalize tangent
            tangent = 1 / 2 * (tangentm1+tangentp1) 
            normal = np.array([-tangent[1], tangent[0]]) # Compute normal unit vector, y component of tangent is the -x component of normal

        # Tangential and normal velocities
        v_tangential = np.dot(velocity_k, tangent) # velocity along tangent
        v_normal = np.dot(velocity_k, normal) # velocity along normal

        ##################### Crude approximation for high Re conditions #####################
        # Tangential and normal forces
        #if mu < 1:
        #    Ft = -0.5*C_t*v_tangential*tangent
        #    Fn = -0.5*C_n*rho_gl*2*np.mean(R[:])*np.linalg.norm(v_normal)*v_normal*normal

        #else:
        ###################################################################################
        Ft = -C_t * v_tangential * tangent # Tangential components of drag
        Fn = -C_n * v_normal * normal # Normal components of drag
        Ftx += Ft[0]*deltaL
        Fnx += Fn[0]*deltaL


        # Add local hydrodynamic force
        Fhydro[2 * k:2 * k + 2] += (Ft + Fn) * deltaL

    return Fhydro, Jhydro, Ftx, Fnx

def RegGreenFunction(r, epsilon):
    r_norm = np.linalg.norm(r)
    denom = (r_norm ** 2 + epsilon ** 2) ** (3 / 2)
    Sij = np.eye(2) * (r_norm ** 2 + 2 * epsilon ** 2) / denom
    for i in range(2):
        for j in range(2):
            Sij[i, j] += r[i] * r[j] / denom
    return Sij

def getSBT(q, q_old, dt, mu, deltaL, R, nv,rho_gl):
    epsilon = deltaL*0.5
    Fregs = np.zeros_like(q)
    vnorm = np.zeros_like(q)
    norm = np.zeros_like(q)
    rhat = np.zeros_like(q)
    Sij = np.zeros(2)
    Sij_bar = np.zeros((len(q), len(q)))
    velocities = (q - q_old) / dt  # Compute velocities directly [2N]

    for j in range(nv):

        velocities[2 * j:2 * j + 2] = velocities[2 * j:2 * j + 2] * R[j] / R[0] # Radius proportional velocity for stokeslet
        xj, yj = q[2 * j], q[2 * j + 1]
        uj = velocities[2 * j:2 * j + 2]  # Velocity vector of particle j [2N]
        vx, vy = (q[2 * j] - q_old[2 * j]) / dt, (q[2 * j + 1] - q_old[2 * j + 1]) / dt # Compute velocity
        velocity_j = np.array([vx, vy])
        if j == nv - 1:
            xjm1, yjm1 = q[2 * (j - 1)], q[2 * (j - 1) + 1] # Store prev node
            tangent = np.array([xj - xjm1, yj - yjm1]) # Compute tangent vector as direction of segnment
            tangent /= np.linalg.norm(tangent) # Normalize tangent
            normal = np.array([-tangent[1], tangent[0]])
        elif j == 0:
            xjp1, yjp1 = q[2 * (j + 1)], q[2 * (j + 1) + 1]  # Store prev node
            tangent = np.array([xjp1 - xj, yjp1 - yj]) # Compute tangent vector as direction of segnment
            tangent /= np.linalg.norm(tangent) # Normalize tangent
            normal = np.array([-tangent[1], tangent[0]])
        else:
            # Local tangent and normal
            xjp1, yjp1 = q[2 * (j + 1)], q[2 * (j + 1) + 1] # Store next node
            xjm1, yjm1 = q[2 * (j - 1)], q[2 * (j - 1) + 1] # Store next node
            tangentp1 = np.array([xjp1 - xj, yjp1 - yj]) # Compute tangent vector as direction of segnment
            tangentp1 /= np.linalg.norm(tangentp1) # Normalize tangent
            tangentm1 = np.array([xj - xjm1, yj - yjm1]) # Compute tangent vector as direction of segnment
            tangentm1 /= np.linalg.norm(tangentm1) # Normalize tangent
            tangent = 1 / 2 * (tangentm1+tangentp1) 
            normal = np.array([-tangent[1], tangent[0]]) # Compute normal unit vector, y component of tangent is the -x component of normal
        
        # Tangential and normal velocities
        v_tangential = np.dot(velocity_j, tangent) # velocity along tangent
        v_normal = np.dot(velocity_j, normal) # velocity along normal
        vnorm[2*j] = velocity_j[0]*normal[0]
        vnorm[2*j+1] = velocity_j[1]*normal[1]
        norm[2*j] = normal[0]
        norm[2*j+1] = normal[1]
        ##################### Crude approximation for high Re conditions #####################
        #if mu < 1:
            #Fn = -4*np.pi*mu*2*np.mean(R[:])*np.linalg.norm(v_normal)*v_normal*normal*rho_gl
            #Fregs[2 * j:2 * j + 2] += (Fn) * deltaL
        #else:
        ######################################################################################
        Fn = -4*np.pi*mu*v_normal*normal
        Fregs[2 * j:2 * j + 2] += (Fn) * deltaL


        for n in range(nv):
            if j == n:
                continue

            xn, yn = q[2 * n], q[2 * n + 1] # [2N]
            rx = xj - xn 
            ry = yj - yn # [2N]
            r = np.array([rx,ry])
            rnorm = np.linalg.norm([rx,ry])
            rhat[2*n] = r[0] / rnorm
            rhat[2*n+1] = r[1] / rnorm
            
            Sij = RegGreenFunction(r, epsilon) # [2 x 2]
            Sij_bar[2 * j:2 * j + 2, 2 * n:2 * n + 2] = Sij
            
            # Solve for force contribution
 ####################### Crude approximation for high Re conditions #####################     
   #if mu < 1e-4:
   #    Fregs += -np.dot((np.linalg.solve(Sij_bar, velocities) * (8 * np.pi * mu)),rhat)*rhat*np.linalg.norm(velocities)*np.mean(R[:])*rho_gl
   #else:
 ######################################################################################
    Fregs += -np.dot((np.linalg.solve(Sij_bar, velocities) * (8 * np.pi * mu)),rhat)*rhat 
    return Fregs


# Add Resistive Force Theory (RFT) to objfun
# Add hydrodynamic force contribution to objfun
# Add hydrodynamic force contributions based on SBT to objfun
def objfun(q_guess, q_old, u_old, dt, tol, maximum_iter,
           m, mMat,  # inertia
           EI, EA,   # elastic stiffness 
           deltaL,
           free_index,
           RodLength, R, SBTFlag
           ): 


    q_new = q_guess.copy()
    iter_count = 0
    error = tol * 10
    flag = 1

    # Initialize array to store hydrodynamic force magnitudes
    hydrodynamic_force_magnitudes = np.zeros(nv)
    axial_force_magnitudes = np.zeros(nv)

    while error > tol:
        Fb, Jb = getFb(q_new, EI, deltaL, R)
        Fs, Js = getFs(q_new, EA, deltaL, R)

## Need If statement for using RFT vs SBT and to compute SBT using equation 16 from helicoil PNAS paper
        if SBTFlag == 0:
            Fhydro, Jhydro, Ftx, Fnx = getRFT(q_new, q_old, dt, R, nv, rho_gl)
        elif SBTFlag == 1:
            Fhydro = getSBT(q_new, q_old, dt, mu, deltaL, R, nv, rho_gl)
            Jhydro = np.zeros([len(q), len(q)])
        else:
            print(f'SBTflag input is invalid, please set to 1 for SBT or 0 for RFT')

        # Calculate SBT-based hydrodynamic forces
        for k in range(nv):  # Skip the first and last nodes

            # Store the magnitude of the hydrodynamic force for color mapping
            hydrodynamic_force_magnitudes[k] = np.linalg.norm(Fhydro[2*k:2*k+2])
            axial_force_magnitudes[k] = np.linalg.norm(Fs[2*k:2*k+2])

        # Equation of motion with hydrodynamic forces
        f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fb + Fs  + Fhydro)

        # Jacobian manipulation
        J = mMat / dt**2 - (Jb + Js + Jhydro)

        f_free = f[free_index]
        J_free = J[np.ix_(free_index, free_index)]

        dq_free = np.linalg.solve(J_free, f_free)
        q_new[free_index] = q_new[free_index] - dq_free

        error = np.linalg.norm(f_free)
        iter_count += 1
        print(f'Iter={iter_count-1}, error={error:.6e}')

        if iter_count > maximum_iter:
            flag = -1
            return q_new, flag, hydrodynamic_force_magnitudes, axial_force_magnitudes, Ftx, Fnx
    if SBTFlag == 0:
        return q_new, flag, hydrodynamic_force_magnitudes, axial_force_magnitudes, Ftx, Fnx
    else:
        return q_new, flag, hydrodynamic_force_magnitudes, axial_force_magnitudes

# Inputs and parameters
SBTFlag = 0 # Set to 0 for RFT, 1 for SBT
nv = 30  # Number of vertices
RodLength = 0.1
deltaL = RodLength / (nv-1)
dt = 1e-3

#nv = round(RodLength / deltaL) + 1  # Number of vertices
ndof = 2 * nv

#deltaL = RodLength / (nv - 1)
all_DOFs = np.arange(ndof)
free_index = all_DOFs

# Set head motion parameters
head_amplitude = 0.05
head_frequency = 2.83
isun = 1 # Set to 0 for nonuniform, 1 for uniform

# Radius of nodes, material properties, etc.
free_index = all_DOFs
Head_x = 0.0
Head_index = np.array([0, int(np.round(Head_x / deltaL))])

# Cross-sectional radius of rod
ro = 1e-2

# Radius of spheres
R = np.zeros(nv)  # Vector of size N - Radius of N nodes
R[:] = ro
midNode = int((nv + 1) / 2)
Vtotal = 0
roavg = 0


for g in range(nv):
    if isun == 0:
        R[g] = ro * (((nv - 0.19 * g) / nv) ** 9)
        unstr = "NUn"
    else:
        isun = 1
        unstr = "Un"
    Vtotal += 4/3 * np.pi * R[g] ** 3
    roavg += R[g]
roavg /= nv

# Densities
rho_metal = 2700*(RodLength*np.pi*ro**2) / (Vtotal) # Normalize Density to that of a Al cylinder of radius ro and length RL
rho_gl = 1000
rho = rho_metal

# Young's modulus, Al
Y = 7e10

# Viscosity
mu = 5
visc = mu

# Maximum number of iterations in Newton Solver
maximum_iter = 100

# Total simulation time (it exits after t=totalTime)
totalTime = 2

# Utility quantities
ne = nv - 1
EI = Y * np.pi * (ro ** 4) / 12
EA = Y * np.pi * (ro ** 2) 

# Tolerance on force function
tol = EI / RodLength ** 2 * 1e-5

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


# Initial conditions
q0 = np.zeros(ndof)
for c in range(nv):
    q0[2 * c] = nodes[c, 0]
    q0[2 * c + 1] = nodes[c, 1]

q = q0.copy()
u = (q - q0) / dt

# Number of time steps
Nsteps = round(totalTime / dt)
#psize = 2
nframes = (totalTime / (2*head_frequency*dt)) 
#nframes = 50
shift = nframes * dt / 2
#plotStep = Nsteps / nframes 
plotStep = 1
ctime = 0

# Initialize arrays to store results for plotting
all_pos = np.zeros(Nsteps)
all_v = np.zeros(Nsteps)
all_Ftx = np.zeros(Nsteps)
all_Fnx = np.zeros(Nsteps)
all_Fx = np.zeros(Nsteps)
midAngle = np.zeros(Nsteps)
head_x_positions = np.zeros(int(Nsteps / plotStep))
head_x_velocity = np.zeros(int(Nsteps / plotStep))
Ftnet = np.zeros(int(Nsteps / plotStep))
Fnnet = np.zeros(int(Nsteps / plotStep))
Fxnet = np.zeros(int(Nsteps / plotStep))
t = np.zeros(int(Nsteps / plotStep))
tshift = np.zeros(int(Nsteps / plotStep))

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
    global q, q0, u, ctime, plotStep
    print(f't={ctime:.6f}')

    # Apply head motion
    y_head, vy_head = apply_head_motion(ctime, head_amplitude, head_frequency)
    q[1] = y_head
    u[1] = vy_head

    # Set plot limits based on initial data (adjust as needed)
    ax1.set_xlim(np.mean(q[::2]) - 3*RodLength/4, np.mean(q[::2]) + 3*RodLength/4)
    ax1.set_ylim(-1, 1)
    ax2.set_xlim(np.mean(q[::2]) - 3*RodLength/4, np.mean(q[::2]) + 3*RodLength/4)
    ax2.set_ylim(-1, 1)

    # Solve for new configuration with updated head motion
    if SBTFlag == 0:
        q, error, hydrodynamic_force_magnitudes, axial_force_magnitudes, Ftx, Fnx = objfun(
            q, q0, u, dt, tol, maximum_iter, m, mMat, EI, EA, deltaL, free_index, 
            RodLength, R, SBTFlag)
    else:
        q, error, hydrodynamic_force_magnitudes, axial_force_magnitudes = objfun(
            q, q0, u, dt, tol, maximum_iter, m, mMat, EI, EA, deltaL, free_index, 
            RodLength, R, SBTFlag)

    if error < 0:
        print('Could not converge. Stopping animation.')
        ani.event_source.stop()
        return

    # Update velocities and positions for the next step
    u = (q - q0) / dt
    ctime += dt
    q0 = q

    all_pos[timeStep] = q[0]
    all_v[timeStep] = u[0]

    #if (timeStep) % (plotStep) == 0:
        #n = (timeStep) / plotStep
    n=0
    head_x_positions[int(timeStep / plotStep)] = np.mean(all_pos[0:timeStep+1])
    head_x_velocity[int(timeStep / plotStep)] = np.mean(all_v[0:timeStep+1])
    if SBTFlag == 0:
        all_Ftx[timeStep] = Ftx
        all_Fnx[timeStep] = Fnx
        all_Fx[timeStep] = Ftx + Fnx
        Ftnet[int(timeStep / plotStep)] = np.mean(all_Ftx[int(n*(plotStep)):timeStep+1])
        Fnnet[int(timeStep / plotStep)] = np.mean(all_Fnx[int(n*(plotStep)):timeStep+1])
        Fxnet[int(timeStep / plotStep)] = np.mean(all_Fx[int(n*(plotStep)):timeStep+1])
    t[int(timeStep / plotStep)] = timeStep * dt

    # Normalize force magnitudes for color mapping
    normalized_hydro_magnitudes = hydrodynamic_force_magnitudes / np.max(hydrodynamic_force_magnitudes)
    normalized_axial_magnitudes = axial_force_magnitudes / np.max(axial_force_magnitudes)

    # Extract x and y positions
    x1 = q[::2]
    x2 = q[1::2]

    # Node sizes proportional to their radius
    sizes = (R / np.max(R)) * 200  # Scale sizes for better visualization

    # Update line to connect scatter points
    line1.set_data(x1, x2)
    line2.set_data(x1, x2)

    # Update scatter plots with new data
    sc1.set_offsets(np.c_[x1, x2])
    sc1.set_array(normalized_hydro_magnitudes)
    sc1.set_sizes(sizes)  # Update node sizes

    sc2.set_offsets(np.c_[x1, x2])
    sc2.set_array(normalized_axial_magnitudes)
    sc2.set_sizes(sizes)  # Update node sizes

    # Update titles with current time
    ax1.set_title(f'Hydrodynamic Force Magnitude at t={ctime:.6f}')
    ax2.set_title(f'Axial Force Magnitude at t={ctime:.6f}')

    return sc1, sc2, line1, line2


if SBTFlag == 0:
    suff = f'RFTv6_RL{RodLength}_mu{mu}_N{nv}_dt{dt}_HA{head_amplitude}_HF{head_frequency}_{unstr}'
elif SBTFlag == 1:
    suff = f'SBTv6_RL{RodLength}_mu{mu}_N{nv}_dt{dt}_HA{head_amplitude}_HF{head_frequency}_{unstr}'
else:
    print(f'SBTflag input is set to {SBTFlag} - this is invalid, please set to 1 for SBT or 0 for RFT')



# Run animation
ani = FuncAnimation(fig, update_plot, frames=Nsteps, interval = 50, blit=False, repeat=False)
#plt.show()
# Save the animation
ani.save(f'{suff}.gif', writer='ffmpeg', fps=30)
plt.close()

# Plot results
#head_x_velocity_avg = pd.Series(head_x_velocity).rolling(window=round(totalTime / dt / plotStep)).mean()
Re = rho_gl*np.abs(head_x_velocity)*RodLength/mu
#t = np.linspace(0, totalTime, plotStep)

# New plot for head x-position vs. time
plt.figure(5)
plt.plot(t, head_x_positions, 'b')
plt.xlabel('Time, t [s]')
plt.ylabel('Head X Position, x [m]')
plt.title(f'X-Position ({suff})')
#plt.ylim([-0.35,0.01])
plt.savefig(f'head_x_position_{suff}.png')

# New plot for head x-velocity vs. time
plt.figure(6)
plt.plot(t, head_x_velocity, 'b')
plt.xlabel('Time, t [s]')
plt.ylabel('Head X Velocity, x [m]')
plt.title(f'X-Velocity ({suff})')
plt.savefig(f'head_x_velocity_{suff}.png')

if SBTFlag == 0:
    # New plot for Drag from Tangent of RFT sim
    plt.figure(7)
    plt.plot(t, Ftnet, 'b')
    plt.xlabel('Time, t [s]')
    plt.ylabel('Drag from Tangent [N]')
    plt.title(f'Tangential Drag({suff})')
    plt.savefig(f'Dragt{suff}.png')

    # New plot for Thrust from Normal of RFT sim
    plt.figure(8)
    plt.plot(t, Fnnet, 'b')
    plt.xlabel('Time, t [s]')
    plt.ylabel('Thrust from Normal [N]')
    plt.title(f'Normal Thrust ({suff})')
    plt.savefig(f'ThrustN{suff}.png')

    # New plot for Net Thrust of RFT sim
    plt.figure(9)
    plt.plot(t, Fxnet, 'b')
    plt.xlabel('Time, t [s]')
    plt.ylabel('Net Thrust [N]')
    plt.title(f'Net Thrust ({suff})')
    
    plt.savefig(f'NetThrust{suff}.png')

# New plot for head x-velocity vs. time
plt.figure(10)
plt.plot(t, Re, 'b')
plt.xlabel('Time, t [s]')
plt.ylabel('Re')
plt.title(f'Re ({suff})')
plt.savefig(f'Re_{suff}.png')

plt.show()



