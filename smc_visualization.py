import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator



def trace_smc(Traject):
    """
    Process the trajectories obtain from the SMC_squared in a matrix form
    """
    matrix_dict = {}
    stateName=list(Traject[0].columns[1:])
    # Iterate through each state name
    for state in stateName:
      # Extract matrices for each state from all dataframes
      state_matrices = [df[state].values.reshape(1, -1) for df in Traject]
      # Concatenate matrices horizontally
      combined_matrix = np.concatenate(state_matrices, axis=1)
      # Reshape the combined matrix based on the shape of the original dataframe
      reshaped_matrix = combined_matrix.reshape(-1,Traject[0].shape[0])
          # Store the reshaped matrix in the dictionary with state name as the key
      matrix_dict[state] = reshaped_matrix 
    return matrix_dict


def plot_smc(matrix, ax, separation_point=None, Date=None, window=1):
    """
    Plot the SMC^2 results using Matplotlib.
    
    Parameters:
    matrix: The input matrix where columns represent time steps and rows represent samples.
    ax: The matplotlib axis to plot on.
    col_med: The color for the median line. 
    Date: Optional date array for x-axis. If None, numeric time steps are used.
    smooth: Boolean flag to apply smoothing to the curves.
    window_size: The window size for smoothing (default is 7).
    """
   
    # Replace outliers in the matrix
    matrix = corrected_matrix(matrix)
    
    # Calcupredthe median along the columns (axis=0)
    median_values = np.nanmedian(matrix, axis=0)

    # Calcupredthe 95%, 90%, 75%, and 50% credible intervals
    credible_interval_95 = np.nanpercentile(matrix, [2.5, 97.5], axis=0)
    credible_interval_90 = np.nanpercentile(matrix, [5, 95], axis=0)
    credible_interval_75 = np.nanpercentile(matrix, [12.5, 85.5], axis=0)
    credible_interval_50 = np.nanpercentile(matrix, [25, 75], axis=0)

    median_values = pd.Series(median_values).rolling(window=window, min_periods=1).mean().values
    credible_interval_95 = pd.DataFrame(credible_interval_95).T.rolling(window=window, min_periods=1).mean()
    credible_interval_90 = pd.DataFrame(credible_interval_90).T.rolling(window=window, min_periods=1).mean()
    credible_interval_75 = pd.DataFrame(credible_interval_75).T.rolling(window=window, min_periods=1).mean()
    credible_interval_50 = pd.DataFrame(credible_interval_50).T.rolling(window=window, min_periods=1).mean()

    # Define time steps as either numeric or date-based
    T = matrix.shape[1]
    if Date is not None:
        time_steps = pd.to_datetime(Date)  # Ensure time_steps is in datetime format
    else:
        time_steps = np.arange(T)
    if separation_point is not None:
        condition = time_steps > separation_point
        # Split data for the condition
        time_steps_fitt = time_steps[~condition]
        time_steps_pred= time_steps[condition]
    
        # Plot credible intervals with color changes
        for ci, alpha, label in zip(
            [credible_interval_95, credible_interval_90, credible_interval_75, credible_interval_50],
            [0.08, 0.25, 0.43, 1],  # Transparency levels for the credible intervals
            ['95% CrI', '90% CrI', '75% CrI', '50% CrI']  # Labels for intervals
        ):
            ax.fill_between(time_steps_fitt, ci[0][~condition], ci[1][~condition], color='steelblue', alpha=alpha)
            ax.fill_between(time_steps_pred, ci[0][condition], ci[1][condition], color='mediumpurple', alpha=alpha)
    
        # Plot the median line with dynamic color change
        ax.plot(time_steps_fitt, median_values[~condition], color='midnightblue', lw=2)
        ax.plot(time_steps_pred, median_values[condition], color='purple', lw=2)
    
        # Add a vertical dashed line at the separation point
        ax.axvline(separation_point, color='k', linestyle='--', lw=2)
    else:
        for ci, alpha, label in zip(
            [credible_interval_95, credible_interval_90, credible_interval_75, credible_interval_50],
            [0.08, 0.25, 0.43, 1],  # Transparency levels for the credible intervals
            ['95% CrI', '90% CrI', '75% CrI', '50% CrI']  # Labels for intervals
        ):
            ax.fill_between(time_steps, ci[0], ci[1], color='steelblue', alpha=alpha)
            # Plot the median line with dynamic color change
            ax.plot(time_steps, median_values, color='midnightblue', lw=2)

    # Configure x-axis formatting for dates with 3-month grid spacing
    if Date is not None:
        ax.xaxis.set_major_locator(MonthLocator(interval=3))  # Major ticks every 3 months
        ax.xaxis.set_minor_locator(MonthLocator())  # Minor ticks every month
        ax.xaxis.set_major_formatter(DateFormatter('%b %y'))  # Format: "Jan 22"
       
        ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    # Add grid and background color
    ax.grid(True, linestyle='--', alpha=0.9)  # Add grid with dashed lines
    ax.set_facecolor('whitesmoke')  # Add background color for the subplot

    # Add labels and legend
   
def compute_model_average(matrix_dict_dthp, matrix_dict_sir, w_dthp, w_sir):
    matrix_dict_avg = {}
    for key in matrix_dict_dthp.keys():
        if key in matrix_dict_sir.keys():
            matrix_dict_avg[key] = w_dthp * matrix_dict_dthp[key] + w_sir * matrix_dict_sir[key]
    return matrix_dict_avg


def corrected_matrix(matrix):
    # Calcupredthe IQR for each column
    q1 = np.percentile(matrix, 25, axis=0)
    q3 = np.percentile(matrix, 75, axis=0)
    iqr = q3 - q1
    
    # Calcupredthe lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Replace outliers with appropriate values
    for col in range(matrix.shape[1]):
        col_values = matrix[:, col]
        col_non_outliers = col_values[(col_values >= lower_bound[col]) & (col_values <= upper_bound[col])]
        max_non_outlier = np.mean(col_non_outliers)
        min_non_outlier = np.mean(col_non_outliers)
        
        # Replace outliers above the upper bound
        outliers_above = col_values[col_values > upper_bound[col]]
        if len(outliers_above) > 0:
            matrix[col_values > upper_bound[col], col] = max_non_outlier
        
        # Replace outliers below the lower bound
        outliers_below = col_values[col_values < lower_bound[col]]
        if len(outliers_below) > 0:
            matrix[col_values < lower_bound[col], col] = min_non_outlier
    
    return matrix
