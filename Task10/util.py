import seaborn as sns
import matplotlib.pyplot as plt

sns.set(
    context="notebook",
    style="whitegrid",
    rc={"figure.dpi": 120, "scatter.edgecolors": "k"},
)


def plot_vector(u, v):
    props = dict(arrowstyle="->", color="r", linewidth=3, shrinkA=0, shrinkB=0)
    plt.gca().annotate("", v, u, arrowprops=props)


# Plot a set of digits
def plot_digits(data, title):
    fig, axes = plt.subplots(
        4,
        10,
        figsize=(10, 4),
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            data[i].reshape(8, 8), cmap="binary", interpolation="nearest", clim=(0, 16)
        )

    plt.suptitle(title)
    plt.show()


# Plot a set of faces
def plot_faces(data, title, h, w):
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(8, 8),
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(h, w), cmap=plt.cm.gray)

    plt.suptitle(title)
    plt.show()


def plot_cumulative_variance(var_explained):
    plt.figure()
    plt.plot(var_explained)
    plt.title("PCA on Digits: Cumulative Variance Explained")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.show()


def plot_projection(X_digits_trans, digits, y_digits):
    plt.figure()
    for label in range(10):
        X_sub = X_digits_trans[y_digits == label]
        plt.scatter(x=X_sub[:, 0], y=X_sub[:, 1], label=digits.target_names[label])
        props = dict(boxstyle="round", facecolor="white", alpha=0.9)
        plt.text(X_sub[:, 0].mean(), X_sub[:, 1].mean(), label, bbox=props)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.title("Digits PCA Projection")
    plt.xlabel("$\hat{X}_0$")
    plt.ylabel("$\hat{X}_1$")
    plt.show()
