from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    m = 1000
    mu, sigma = 10, 1
    X = np.random.normal(mu, sigma, m)
    uni_gaussian = UnivariateGaussian().fit(X)
    print("(" + str(uni_gaussian.mu_) + ", " + str(uni_gaussian.var_) + ")")

    # raise NotImplementedError()

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_distance = []
    for m in ms:
        Y = X[:m + 1]
        estimated_distance.append(np.abs(UnivariateGaussian().fit(Y).mu_ - mu))

    go.Figure([go.Scatter(x=ms, y=estimated_distance, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Estimation Of Absolute Distance Between "
                        r"Estimated And True Value Of Expectation As Function "
                        r"Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$\\text{ - absolute distance}$",
                  height=400)).show()
    # raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model

    # raise NotImplementedError()
    X = np.sort(X)
    pdf_values = uni_gaussian.pdf(X)

    go.Figure([go.Scatter(x=X, y=pdf_values, mode='markers+lines',
                                marker=dict(color="blue"), showlegend=False)]
                    ).update_layout(
        title_text=r"$\text{Plotting Empirical PDF of fitted model}$",
        xaxis_title="$m\\text{ - sample value}$",
        yaxis_title="r$\\text{ - PDF result}$",
        height=400).show()




def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    mu = [0, 0, 4, 0]
    sigma = [[1, 0.2, 0, 0.5],
             [0.2, 2, 0, 0],
             [0, 0, 1, 0],
             [0.5, 0, 0, 1]]
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    multi_gaussian = MultivariateGaussian().fit(samples)
    print(multi_gaussian.mu_)
    print(multi_gaussian.cov_)


    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    likelihood_result = []
    ms = np.linspace(-10, 10, 200)
    for f1 in ms:
        row = []
        for f3 in ms:
            val = multi_gaussian.log_likelihood(np.array([f1, 0, f3, 0]),
                                                np.array(sigma), samples)
            row.append(multi_gaussian.log_likelihood(np.array([f1, 0, f3, 0]),
                                                     np.array(sigma), samples))
        likelihood_result.append(row)
    go.Figure(data=go.Heatmap(x=ms, y=ms, z=likelihood_result)).update_layout(
        title_text=r"$\text{Evaluation of log-likelihood with expectation [f1, 0, f3, 0]}$",
        xaxis_title="$\\text{ f3 values}$",
        yaxis_title="r$\\text{ f1 values}$",
        ).show()
    # Question 6 - Maximum likelihood
    # raise NotImplementedError()
    row, col = 0, 0
    max_num = likelihood_result[0][0]
    for i in range(len(likelihood_result)):
        for j in range(len(likelihood_result[i])):
            if likelihood_result[i][j] > max_num:
                max_num = likelihood_result[i][j]
                row = i
                col = j
    print(format(ms[row], ".3f"))
    print(format(ms[col], ".3f"))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
