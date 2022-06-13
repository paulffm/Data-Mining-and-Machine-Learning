#!/usr/bin/env python
# coding: utf-8
"""

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import util


def fit_pca(X, n_components) -> PCA:
    """Conducts principal component analysis on the given dataset and returns the fitted PCA model."""
    pca = PCA(n_components=2)
    pca.fit(X)
    return pca


def transform_data_by_pc(X, pca) -> [np.ndarray, float, float]:
    """Transforms the data by the principal components and returns the transformed data, as well as the variance for
    the first and second dimension. """
    X = pca.transform(X)
    return X, np.var(X[:, 0]), np.var(X[:, 1])


def plot_principal_components(X, explained_variance, pca_components, mean, title, xlabel, ylabel) -> None:
    """Plots the data set and its principal components as vectors."""
    plt.figure()
    plt.scatter(*X.T, alpha=0.7)
    for length, vector in zip(explained_variance, pca_components):
        print("plot one", vector, length)
        v = vector * 2 * np.sqrt(length)
        util.plot_vector(mean, mean + v)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.show()


def main():
    # Make experiments reproducible
    np.random.seed(0)

    # Generate data
    cov = np.array([[2.0, 2.5], [-0.5, 1.0]])
    X = np.random.randn(200, 2).dot(cov) * 2.5 + 7

    # Plot data
    plt.figure()
    plt.scatter(*X.T)
    plt.xlabel("$X_0$")
    plt.ylabel("$X_1$")
    plt.title("Sample Data")
    plt.axis("equal")
    plt.show()

    # Compute variance for each component
    v0_var = X[:, 0].var()
    v1_var = X[:, 1].var()

    print("Variances with standard basis:")
    print()
    print("Var[X_0] =", round(v0_var / (v0_var + v1_var) * 100) / 100)
    print("Var[X_1] =", round(v1_var / (v0_var + v1_var) * 100) / 100)

    # Run PCA
    pca = fit_pca(X, 2)

    # Plot data
    plot_principal_components(X, pca.explained_variance_, pca.components_, pca.mean_,
                              "Principal Components", "$X_0$", "$X_1$")

    # Project data onto principal components
    X_trans, v0p_var, v1p_var = transform_data_by_pc(X, pca)

    print("Variances with new basis:")
    print()
    print("Var[Xhat_0] =", round(v0p_var / (v0p_var + v1p_var) * 100) / 100)
    print("Var[Xhat_1] =", round(v1p_var / (v0p_var + v1p_var) * 100) / 100)

    # Plot data
    plot_principal_components(X_trans, [v0p_var, v1p_var], np.array([[1, 0], [0, 1]]), [0, 0],
                              "Projected Data", "$\hat{X}_0$", "$\hat{X}_1$")


if __name__ == '__main__':
    main()
