import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate


# Example usage for comparing groups
group_data = {
    "Store A": [50, 60, 55, 65, 70],
    "Store B": [45, 50, 52, 48, 47],
    "Store C": [60, 62, 65, 70, 75]
}

confidence_interval = 0.95
num_samples = 10000  # More samples for better precision


def inverse_gamma_posterior(data):
    """
    Computes and plots the Inverse-Gamma posterior distribution given prior parameters and observed data.
    
    Parameters:
    alpha_prior (float): Prior shape parameter.
    beta_prior (float): Prior scale parameter.
    data (array-like): Observed data points.
    """
    
    alpha_prior, beta_prior = 1, 1  # Non-informative Beta(1,1) prior
    
    n = len(data)
    mean_obs = np.mean(data)
    sum_sq_diff = np.sum((data - mean_obs) ** 2)
    
    # Posterior parameters
    alpha_post = alpha_prior + n / 2
    beta_post = beta_prior + 0.5 * sum_sq_diff
    
    return alpha_prior, beta_prior, alpha_post, beta_post, mean_obs

def probability_of_being_best(results):
    """
    Computes the probability that each group has the highest mean sales using Monte Carlo sampling.
    """
    samples = {}
    
    #normal (Gaussian) distribution X~N(μ,σ^2)
    for group, metrics in results.items():
        samples[group] = np.random.normal(loc=metrics['mean'], scale=np.sqrt(metrics['beta_post'] / (metrics['alpha_post'])), size=num_samples)
    
    best_counts = {group: 0 for group in results}
    
    for i in range(num_samples):
        best_group = max(samples, key=lambda g: samples[g][i])
        best_counts[best_group] += 1
    
    probabilities = {group: count / num_samples for group, count in best_counts.items()}
    return probabilities

def credible_intervals(results, credibility):
    """
    Computes the Bayesian credible interval for each group's mean sales.
    """
    intervals = {}
    for group, metrics in results.items():
        mean_samples = np.random.normal(loc=metrics['mean'], scale=np.sqrt(metrics['beta_post'] / (metrics['alpha_post'])), size=10000)
        lower = np.percentile(mean_samples, (1 - credibility) / 2 * 100)
        upper = np.percentile(mean_samples, (1 + credibility) / 2 * 100)
        intervals[group] = (lower, upper)
    return intervals

def compare_variances(groups):
    """
    Compares the posterior variances of multiple groups using the Inverse-Gamma model.
    """
   
    results = {}
    plt.figure(figsize=(8, 5))
    x = np.linspace(0.01, max([max(data) for data in groups.values()]) * 2, 1000)
    
    for group, data in groups.items():
        alpha_prior, beta_prior, alpha_post, beta_post, mean_obs = inverse_gamma_posterior(np.array(data))
        post_dist = stats.invgamma(alpha_post, scale=beta_post)
        plt.plot(x, post_dist.pdf(x), label=f'{group} (Mean={mean_obs:.2f}, Variance α={alpha_post}, β={beta_post})')
        results[group] = {"mean": mean_obs, "alpha_prior": alpha_prior, "beta_prior": beta_prior, "alpha_post": alpha_post, "beta_post": beta_post}
    
    
    probabilities = probability_of_being_best(results)
    intervals = credible_intervals(results, confidence_interval)

    # ========================
    # Results
    # ========================
    headers = [
        "Variant", "Samples", "Mean", "prior - post","P2BB (Best Probability)", f"Credible Interval ({confidence_interval*100}%)"
    ]

    data = [[
        group, group_data[group], results[group]['mean'],
        f"({ results[group]['alpha_prior']}, {results[group]['beta_prior']})  -  ({results[group]['alpha_post']:.2f}, {results[group]['beta_post']:.2f})",
        "{:.2%}".format(probabilities[group]), f"[{intervals[group][0]:.2f}% , {intervals[group][1]:.2f}%]"
    ] for group in group_data]
    
    print(tabulate(data, headers, tablefmt="grid"))

    plt.xlabel('Variance (σ²)')
    plt.ylabel('Density')
    plt.title('Comparison of Posterior Inverse-Gamma Distributions')
    plt.legend()
    plt.grid()
    plt.show()
    


    '''
    # Ranking stores by performance (Higher mean, lower variance is better)
    ranked = sorted(results.items(), key=lambda x: (-x[1]['mean'], x[1]['beta']/x[1]['alpha']))
    print("Performance Ranking:")
    for rank, (group, metrics) in enumerate(ranked, 1):
        print(f"{rank}. {group}: Mean Sales = {metrics['mean']:.2f}, Estimated Variance = {metrics['beta']/metrics['alpha']:.2f}")
    '''
    
    return results



compare_variances(group_data)