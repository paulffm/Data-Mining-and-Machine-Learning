#!/usr/bin/env python
# coding: utf-8
"""
@author: Steven Lang, Johannes Czech
Machine Learning Group 2020, TU Darmstadt
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import util


def reconstruct_digits(X_digits_trans, pca) -> np.ndarray:
    """Reconstructs and returns the original feature representation using the fitted PCA model."""
    X_digits_rec = pca.inverse_transform(X_digits_trans)
    return X_digits_rec

def compute_cumulative_explained_variance(pca) -> np.ndarray:
    """Computes and returns the normalized cumulative explained variance of the PCA components."""
    '''explained_variance: The amount of variance explained by each of the selected components. 
    The variance estimation uses n_samples - 1 degrees of freedom.'''
    # cumsum addiert Einträge: A= [1, 2, 3, 4, 5, 6,] np.cumsum(A) >> [1, 3, 6, 10, 15, 21]
    var_explained = np.cumsum(pca.explained_variance_)
    var_explained /= var_explained[-1] # var/var(-1)
    return var_explained

def run_pca(X_digits, n_components) -> np.ndarray:
    """Runs PCA with n components on the given data and returns the projected data with the new basis."""
    pca = PCA(n_components)
    return pca.fit_transform(X_digits)

def main():
    # Load data
    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target

    # Run PCA
    X_digits_trans = run_pca(X_digits, 2)

    # Plot data
    util.plot_projection(X_digits_trans, digits, y_digits)

    # Load digits
    X_digits, y_digits = load_digits(return_X_y=True)

    # Run PCA
    pca = PCA(n_components=X_digits.shape[1])
    pca.fit(X_digits)

    # Compute cumulative explained variance
    var_explained = compute_cumulative_explained_variance(pca)

    # Plot cumulative variance
    # 75 % can be explained with 10 components
    util.plot_cumulative_variance(var_explained)

    # Show original data
    util.plot_digits(X_digits, title="Original Digits")

    # Run PCA to obtain compression
    k = 10
    pca = PCA(n_components=k)
    X_digits_trans = pca.fit_transform(X_digits)

    # Reconstruct
    X_digits_reconst = reconstruct_digits(X_digits_trans, pca)

    # Plot reconstruction
    util.plot_digits(
        X_digits_reconst, title=f"Reconstructed Digits after Compression with $k={k}$"
    )

    # Add noise
    X_digits_noisy = X_digits + np.random.randn(*X_digits.shape) * 4

    # Plot noisy digits
    util.plot_digits(X_digits_noisy, title="Noisy Digits")

    # Run PCA
    k = 10
    pca = PCA(n_components=k)
    pca.fit(X_digits_noisy)
    X_digits_trans = pca.transform(X_digits_noisy)

    # Invert the PCA transformation
    X_digits_reconst = reconstruct_digits(X_digits_trans, pca)

    # Plot reconstructed digits
    util.plot_digits(X_digits_reconst, title=f"Reconstructed Digits ($k={k}$)")


if __name__ == '__main__':
    main()
