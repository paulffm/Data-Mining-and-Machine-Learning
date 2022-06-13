#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
import util


def extract_eigenfaces(X_lfw, height, weight, n_components) -> np.ndarray:
    """Extracts the first n_components eigenfaces of the dataset at a given height and width using PCA."""
    pca = PCA(n_components)
    pca.fit(X_lfw)
    eigenfaces = pca.components_.reshape((n_components, height, weight))
    return eigenfaces


def main():
    # Load lfw dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    _, height, weight = lfw_people.images.shape
    X_lfw = lfw_people.data

    # Apply PCA
    eigenfaces = extract_eigenfaces(X_lfw, height, weight, n_components=16)

    util.plot_faces(X_lfw, "Original Faces of LFW", height, weight)
    util.plot_faces(eigenfaces, "16 Principal Components of LFW", height, weight)


if __name__ == '__main__':
    main()
