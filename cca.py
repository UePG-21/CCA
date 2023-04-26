import numpy as np
import pandas as pd


def pow_mat(A: np.ndarray, num: int, den: int = 1) -> np.ndarray:
    """Calculate the power of the matrix

    Parameters
    ----------
    A : np.ndarray
        Given matrix
    num : int
        Numerator of the exponent
    den : int, optional
        Denominator of the exponent, by default 1

    Returns
    -------
    np.ndarray
        Root inverse of a matrix
    """
    if A.shape[0] != A.shape[1]:
        raise Exception("`A` should be symmetric")
    if not (isinstance(num, int) and isinstance(den, int)):
        raise Exception("`num` and `den` should be both integer")
    eigval, eigvec = np.linalg.eig(A)
    if (eigval < 0).any() and (num * den < 0):
        raise Exception("Non positive definite matrix cannot be taken inverse")
    return eigvec @ np.diag(eigval ** (num / den)) @ eigvec.T


def order_eig(
        eigval: np.ndarray, eigvec: np.ndarray, ascending: bool = False
    ) -> tuple[np.ndarray]:
    """Order eigenvalues and their corresponding eigenvectors

    Parameters
    ----------
    eigval : np.ndarray
        Eigenvalues
    eigvec : np.ndarray
        Eigenvectors
    ascending : bool, optional
        Order them in ascending order or not, by default False

    Returns
    -------
    tuple[np.ndarray]
        Eigenvalues, eigenvectors
    """
    idx = eigval.argsort()
    if not ascending:
        idx = idx[::-1]
    eigval, eigvec = eigval[idx], eigvec[:, idx]
    return eigval, eigvec


def _cca_population_1(
        Sig11: np.ndarray, Sig22: np.ndarray, Sig12: np.ndarray
    ) -> tuple[np.ndarray]:
    """Canonical Correlation Analysis for population (Part 1)

    Parameters
    ----------
    Sig11 : np.ndarray
        Covariance matrix of X
    Sig22 : np.ndarray
        Covariance matrix of Y
    Sig12 : np.ndarray
        Covariance matrix of X and Y

    Returns
    -------
    tuple[np.ndarray]
        rho's in descending order (only keep the positive ones)
        a's (as columns in matrix)
        b's (as columns in matrix)
    """
    if not (Sig11.shape[0] == Sig11.shape[1] == Sig12.shape[0]):
        raise Exception("matrix shape invalid for X part")
    if not (Sig22.shape[0] == Sig22.shape[1] == Sig12.shape[1]):
        raise Exception("matrix shape invalid for Y part")
    if (Sig11 != Sig11.T).any() or (Sig22 != Sig22.T).any():
        raise Exception("`Sig11` and `Sig22` should be symmetric")
    
    C = pow_mat(Sig11, -1, 2) @ Sig12 @ pow_mat(Sig22, -1, 2)
    eigval, e = order_eig(*np.linalg.eigh(C @ C.T))
    rho = np.sqrt(eigval)
    f = C.T @ e @ np.diag(1 / rho)
    a = pow_mat(Sig11, -1, 2) @ e
    b = pow_mat(Sig22, -1, 2) @ f

    # keep positive rho only
    pos_num = (eigval > 0).sum()
    rho = rho[:pos_num]
    a = a[:, :pos_num]
    b = b[:, :pos_num]

    return rho, a, b


def _cca_population_2(
        mu1: np.ndarray, mu2: np.ndarray, rho: np.ndarray, a: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray]:
    """Canonical Correlation Analysis for population (Part 2)

    Parameters
    ----------
    mu1 : np.ndarray
        Mean vector of X
    mu2 : np.ndarray
        Mean vector of Y
    rho : np.ndarray
        Positive rho's, result from CCA_1()
    a : np.ndarray
        Weights of X to get U, result from CCA_1()
    b : np.ndarray
        Weights of Y to get V, result from CCA_1()

    Returns
    -------
    tuple[np.ndarray]
        Mean vector of (U, V)
        Covariance matrix (U, V)
    """
    if mu1.shape[0] != a.shape[0]:
        raise Exception("shape of `mu1` is wrong")
    if mu2.shape[0] != b.shape[0]:
        raise Exception("shape of `mu2` is wrong")
    
    E_U = a.T @ mu1.reshape((-1, 1))
    E_V = b.T @ mu2.reshape((-1, 1))
    E = np.vstack((E_U, E_V)).flatten()
    
    r = rho.shape[0]
    Var_UU, Var_VV = np.eye(r), np.eye(r)
    Var_UV, Var_VU = np.diag(rho), np.diag(rho)
    Var = np.block([[Var_UU, Var_UV], [Var_VU, Var_VV]])

    return E, Var


def cca_population(
        mu1: np.ndarray,
        mu2: np.ndarray,
        Sig11: np.ndarray,
        Sig22: np.ndarray,
        Sig12: np.ndarray
    ) -> dict[str, np.ndarray]:
    """Conanical Correlation Analysis for population
        - Part 1:
        Find corr(U, V) = rho from large to small, where U = a.T @ X, V = b.T @ Y,
        which is equivalent to maximize rho = a.T @ Sig12 @ b under constrains
        Var(U) = a.T @ Sig11 @ a = 1 and Var(V) = b.T @ Sig22 @ b = 1
        - Part 2:
        Calculate E[(U, V)] and Var[(U, V)], where (U, V) = (U1, ... Ur, V1, ... Vr).T
        is a column vector of 2r random variables (r is the number of positive rho's)

    Parameters
    ----------
    mu1 : np.ndarray
        Mean vector of X
    mu2 : np.ndarray
        Mean vector of Y
    Sig11 : np.ndarray
        Covariance matrix of X
    Sig22 : np.ndarray
        Covariance matrix of Y
    Sig12 : np.ndarray
        Covariance matrix of X and Y

    Returns
    -------
    dict[str, np.ndarray]
        rho's in descending order (only keep the positive ones)
        a's (as columns in matrix)
        b's (as columns in matrix)
        Mean vector of (U, V)
        Covariance matrix (U, V)
    """
    rho, a, b = _cca_population_1(Sig11, Sig22, Sig12)
    E, Var = _cca_population_2(mu1, mu2, rho, a, b)
    return {"rho": rho, "a": a, "b": b, "E": E, "Var": Var}


def cca_sample(
        X: pd.DataFrame, Y: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Conanical Correlation Analysis for sample
        - Part 1:
        Find corr(U, V) = rho from large to small, where U = a.T @ X, V = b.T @ Y,
        which is equivalent to maximize rho = a.T @ Sig12 @ b under constrains
        Var(U) = a.T @ Sig11 @ a = 1 and Var(V) = b.T @ Sig22 @ b = 1
        - Part 2:
        Calculate E[(U, V)] and Var[(U, V)], where (U, V) = (U1, ... Ur, V1, ... Vr).T
        is a column vector of 2r random variables (r is the number of positive rho's)

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix X, n samples * p features
    Y : pd.DataFrame
        Design matrix Y, m samples * q features

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]
        dict[str, np.ndarray]
            "rho": rho's in descending order (only keep the positive ones)
            "a": a's (as columns in matrix)
            "b": b's (as columns in matrix)
            "E": Mean vector of (U, V)
            "Var": Covariance matrix (U, V)
        dict[str, pd.DataFrame]
            "U": U's (as columns in matrix)
            "V": V's (as columns in matrix)
    """
    p, q = X.shape[1], Y.shape[1]
    mu1, mu2 = X.mean(), Y.mean()
    Sig = pd.concat([X, Y], axis=1).cov()
    Sig11, Sig22, Sig12 = Sig.iloc[:p, :p], Sig.iloc[-q:, -q:], Sig.iloc[:p, -q:]

    # convert to ndarray
    mu1, mu2 = mu1.values, mu2.values
    Sig11, Sig22, Sig12 = Sig11.values, Sig22.values, Sig12.values

    # U and V (can be calculate when we have sample)
    res = cca_population(mu1, mu2, Sig11, Sig22, Sig12)
    U = pd.DataFrame((X @ res["a"]).values, index=X.index)
    V = pd.DataFrame((Y @ res["b"]).values, index=Y.index)
    
    return res, {"U": U, "V": V}
    