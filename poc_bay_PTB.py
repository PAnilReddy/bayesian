import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


'''
variants = {
    "Blue button": {"conversions": 0, "trials": 0, "prevConversions": 9000, "prevTrials": 10000},
    "Red button": {"conversions": 0, "trials": 0},
    "Green button": {"conversions": 0, "trials": 0}
}
'''
variants = {
    "Blue button": {"conversions": 1500, "trials": 8500, "prevConversions": 90, "prevTrials": 100},
    "Red button": {"conversions": 1450, "trials": 8500},
    "Green button": {"conversions": 2800, "trials": 8500}
}


confidence_interval = 0.95
relative_diff_wrt = "C"  # Baseline variant for comparison
#alpha_prior, beta_prior = 1, 1  # Non-informative Beta(1,1) prior
num_samples = 50000  # More samples for better precision


posteriors = {}
samples = {}
priors = {}

for variant, data in variants.items():
    alpha_prior, beta_prior = 1, 1  
    if "prevConversions" in data:
        alpha_prior = alpha_prior + data["prevConversions"]
        beta_prior = beta_prior + data["prevTrials"] - data["prevConversions"]

    alpha_post = alpha_prior + data["conversions"]
    beta_post = beta_prior + (data["trials"] - data["conversions"])
    
    priors[variant] = {"alpha_prior":alpha_prior, "beta_prior": beta_prior, "alpha_post": alpha_post, "beta_post":beta_post}
    posteriors[variant] = stats.beta(alpha_post, beta_post)
    samples[variant] = posteriors[variant].rvs(num_samples)  # Monte Carlo sampling

# ========================
# Compute Probabilities & Credible Intervals
# ========================
prob_best = {v: np.mean(samples[v] > np.max([samples[x] for x in variants if x != v], axis=0)) for v in variants}
credible_intervals = {v: tuple(100 * x for x in posteriors[v].interval(confidence_interval)) for v in variants}
conv_rates = {
    v: (variants[v]["conversions"] / variants[v]["trials"] if variants[v]["trials"] > 0 else 0)
    for v in variants
}


# ========================
# Results
# ========================
headers = [
    "Variant", "Samples", "Conversions", "Conv. Rate", "prior - post",
    "P2BB (Best Probability)", f"Credible Interval ({confidence_interval*100}%)"
]

#print(variants);
data = [[
    v, variants[v]["trials"], variants[v]["conversions"], "{:.2%}".format(conv_rates[v]),
    f"({priors[v]['alpha_prior']}, {priors[v]['beta_prior']})  -  ({priors[v]['alpha_post']}, {priors[v]['beta_post']})",
    "{:.2%}".format(prob_best[v]), f"[{credible_intervals[v][0]:.2f}% , {credible_intervals[v][1]:.2f}%]"
] for v in variants]

print(tabulate(data, headers, tablefmt="grid"))

# ========================
# Charts (Conversion Rate)
# ========================
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot posterior distributions
x = np.linspace(0, 1, 1000)
for variant, posterior in posteriors.items():
    sns.kdeplot(posterior.rvs(num_samples), label=f"Posterior {variant}", fill=True, ax=ax1)

ax1.set_xlabel("Conversion Rate")
ax1.set_ylabel("Density")
ax1.set_title("Posterior Distributions of Conversion Rates")
ax1.legend()

plt.show()



# ========================
# Export Results to JSON
# ========================
bayesian_data = {
    "variants": list(variants.keys()),
    "posterior_samples": {v: samples[v].tolist() for v in variants},  # Bayesian samples as list
    "credible_intervals": credible_intervals,
    "prob_best": prob_best,
}

#print(f"bayesian_data as response for javascript: {bayesian_data}")


'''

        fetch("bayesian_results.json")
            .then(response => response.json())
            .then(data => {
                const variants = data.variants;
                const revenues = data.revenues;

                // ========================
                // Plot Posterior Distributions
                // ========================
                const posteriorChart = new Chart(document.getElementById("posteriorChart"), {
                    type: "line",
                    data: {
                        labels: Array.from({ length: 1000 }, (_, i) => i / 1000), // X-axis (Conversion Rates)
                        datasets: variants.map((variant, index) => ({
                            label: `Posterior ${variant}`,
                            borderColor: ["blue", "orange", "green"][index],
                            fill: false,
                            data: data.posterior_samples[variant].slice(0, 1000) // Take first 1000 points
                        }))
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: { title: { display: true, text: "Conversion Rate" } },
                            y: { title: { display: true, text: "Density" } }
                        }
                    }
                });

                // ========================
                // Plot Revenue Comparison
                // ========================
                const revenueChart = new Chart(document.getElementById("revenueChart"), {
                    type: "bar",
                    data: {
                        labels: variants,
                        datasets: [{
                            label: "Total Revenue ($)",
                            backgroundColor: ["blue", "orange", "green"],
                            data: Object.values(revenues)
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: { title: { display: true, text: "Revenue ($)" } }
                        }
                    }
                });
            });
            
'''


