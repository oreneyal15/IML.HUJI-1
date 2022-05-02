from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        # raise NotImplementedError()
        features, samples = load_dataset("../datasets/" + f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        # raise NotImplementedError()
        def record_loss(p, X, y):
            losses.append(p.loss(features, samples))
        Perceptron(callback=record_loss).fit(features, samples)

        # Plot figure of loss as function of fitting iteration
        # raise NotImplementedError()
        num_iter = [i for i in range(1, len(losses)+1)]
        go.Figure([go.Scatter(x=num_iter, y=losses, mode='markers+lines')],
                  layout=go.Layout(
                      title="Figure of loss as function of fitting iteration when the space is " + n,
                      xaxis_title="$\\text{number of iteration}$",
                      yaxis_title="r$\\text{loss value}$",

                      )).show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        # raise NotImplementedError()
        X, y = load_dataset("../datasets/" + f)
        # Fit models and predict over training set
        # raise NotImplementedError()
        lda_model = LDA().fit(X, y)
        bayes_model = GaussianNaiveBayes().fit(X, y)
        lda_prediction = lda_model.predict(X)
        bayes_prediction = bayes_model.predict(X)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        # raise NotImplementedError()
        sub_titles = [rf"$\text{{Gaussian Naive Bayes Classifier, With Accuracy = {accuracy(y, bayes_prediction)}}} $",
                      rf"$\text{{LDA Classifier, With Accuracy = {accuracy(y, lda_prediction)}}} $"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=sub_titles)
        plots = []
        lda_index = 1
        gnb_index = 0
        for classifier_prediction, location, name in [(bayes_prediction, (1,1), "Gaussian Naive Bayes"), (lda_prediction, (1, 2), "LDA")]:
            plots.append([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color=classifier_prediction, symbol=y))])
        # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()
        plots[gnb_index].append(go.Scatter(x=bayes_model.mu_[:, 0], y=bayes_model.mu_[:, 1], mode="markers", marker=dict(color="black", symbol="x"), showlegend=False))
        plots[lda_index].append(go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1], mode="markers", marker=dict(color="black", symbol="x"), showlegend=False))

        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()
        for i in range(bayes_model.mu_.shape[0]):
            plots[gnb_index].append(get_ellipse(bayes_model.mu_[i], np.diag(bayes_model.vars_[i])))
        for i in range(lda_model.mu_.shape[0]):
            plots[lda_index].append(get_ellipse(lda_model.mu_[i], lda_model.cov_))
        fig.add_traces(plots[gnb_index], rows=1, cols=1)
        fig.add_traces(plots[lda_index], rows=1, cols=2)

        fig.update_layout(title=f"Using {f} Data Set\n", margin=dict(t=100), yaxis_title="feature 2", xaxis_title="feature 1",
                          yaxis2_title="feature 2", xaxis2_title="feature 1", font_size=15)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
