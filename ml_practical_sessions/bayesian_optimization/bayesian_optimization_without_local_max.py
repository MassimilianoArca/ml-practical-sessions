
import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from mpl_toolkits.mplot3d import Axes3D

# === FUNCTIONS ===

def objective_function(pH, temperature):
    """
    Function with a global maximum in the range:
    pH ∈ [3, 11], temperature ∈ [0, 20].
    The global maximum is at pH = 11, temperature = 2.
    """
    return 80 * np.exp(-((pH-11)**2 + (temperature-2)**2) / 10) + 10 * np.sin(0.5 * pH) + 5 * np.sin(0.5 * temperature)

def target_function(pH, temperature):
    
    result = objective_function(pH, temperature) + np.random.normal(0, 2)  # Add noise
    
    return np.clip(result, 0, 100)  # Clip to the range [0, 100]


# posterior function to compute the mean and standard deviation of the Gaussian Process
def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

# Function to plot the Gaussian Process and acquisition function
def plot_gp(optimizer, pH, Temperature):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontsize=30
    )
    
    gs = plt.GridSpec(2, 1, height_ratios=[2, 2]) # type: ignore
    axis = plt.subplot(gs[0], projection='3d')
    acq = plt.subplot(gs[1], projection='3d')
    
    # get the points that have been evaluated until now
    pH_obs = np.array([[res["params"]["pH"]] for res in optimizer.res])
    Temperature_obs = np.array([[res["params"]["temperature"]] for res in optimizer.res])
    efficiency_obs = np.array([res["target"] for res in optimizer.res])

    # Fit the Gaussian Process with the updated observations
    # This needs to be done because the Gaussian Process is not updated automatically (package instruction)
    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)
    
    # compute the mean and standard deviation of the Gaussian Process
    mu, sigma = posterior(optimizer, np.c_[pH.ravel(), Temperature.ravel()])
    mu = mu.reshape(pH.shape)
    sigma = sigma.reshape(pH.shape)

    # Plot the Gaussian Process mean and the 95% credible interval
    axis.plot_surface(pH, Temperature, mu, color='royalblue', alpha=0.8) # type: ignore
    axis.plot_surface(pH, Temperature, mu + 1.96 * sigma, color='blue', alpha=0.5) # type: ignore
    axis.plot_surface(pH, Temperature, mu - 1.96 * sigma, color='royalblue', alpha=0.5) # type: ignore
    
    # Plot the points that have been evaluated
    axis.scatter(pH_obs.flatten(), Temperature_obs.flatten(), efficiency_obs, color='r', s=50, label='Observations') # type: ignore
    axis.set_xlabel('pH')
    axis.set_ylabel('Temperature')
    axis.set_zlabel('Efficiency') # type: ignore
    axis.set_title('Gaussian Process')
    
    axis.legend()
    
    # Get the acquisition function values
    acq_function = optimizer.acquisition_function
    acq_values = -1 * acq_function._get_acq(gp=optimizer._gp)(np.c_[pH.ravel(), Temperature.ravel()])
    acq_values = acq_values.reshape(pH.shape)

    acq.plot_surface(pH, Temperature, acq_values, cmap='viridis', alpha=0.9) # type: ignore
    
    # plot the max point of the acquisition function
    max_acq = np.max(acq_values)
    max_acq_idx = np.argmax(acq_values)
    max_acq_pH = pH.ravel()[max_acq_idx]
    max_acq_temp = Temperature.ravel()[max_acq_idx]
    acq.scatter(max_acq_pH, max_acq_temp, max_acq, color='gold', s=100, edgecolor='k', linewidth=1.5, label='Next Best Guess') # type: ignore

    acq.set_xlabel('pH')
    acq.set_ylabel('Temperature')
    acq.set_zlabel('Utility') # type: ignore
    
    acq.legend()
    

    acq.set_title('Acquisition Function')

    plt.tight_layout()
    plt.show()
    
  
  
if __name__ == '__main__':
    # Define the pH and temperature ranges
    pH_range = np.linspace(3, 11, 100)
    temperature_range = np.linspace(0, 20, 100)
    pH_mesh, temperature_mesh = np.meshgrid(pH_range, temperature_range)
    
    # Create the optimizer, where 
    # - the target function is the one with noise that we want to maximize
    # - the pH and temperature are the parameters to optimize
    # - the acquisition function is the Upper Confidence Bound
    optimizer = BayesianOptimization(
        f=target_function,
        pbounds={'pH': (3, 11), 'temperature': (0, 20)},
        random_state=27,
        acquisition_function=acquisition.UpperConfidenceBound(random_state=27),
    )
    # First, we will evaluate the target function at two random points
    optimizer.maximize(init_points=2, n_iter=0)
    
    # run the optimization process for 10 iterations
    for _ in range(10):
        optimizer.maximize(init_points=0, n_iter=1)
        plot_gp(optimizer, pH_mesh, temperature_mesh)
    
    # Print the best parameters found and the efficiency at those parameters
    print("Stopped at iteration", len(optimizer.space))
    best_pH = optimizer.max['params']['pH'] # type: ignore
    best_temperature = optimizer.max['params']['temperature'] # type: ignore
    best_efficiency = optimizer.max['target'] # type: ignore
    print(f"Best parameters found: pH={best_pH}, temperature={best_temperature}")
    print(f"Efficiency at best parameters: {best_efficiency}")