import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Set plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

# Path to data directory
DATA_DIR = 'data'
OUTPUT_DIR = 'analysis_results'

# Set up matplotlib style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_dataframe(df, date_column='date'):
    """
    Convert index to datetime and set it as index
    """
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def calculate_daily_returns(df, value_column):
    """
    Calculate daily returns from a value column
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate daily returns
    df_copy['returns'] = df_copy[value_column].pct_change()
    
    # Drop NaN values (first row)
    return df_copy['returns'].dropna()

def load_strategy_returns(strategy_files, data_dir='data'):
    """
    Load strategy returns from pickle files based on provided file list
    
    Parameters:
    -----------
    strategy_files : list of tuples
        List of tuples containing (file_name, strategy_name, value_column)
    data_dir : str, optional
        Directory containing the pickle files, default is 'data'
        
    Returns:
    --------
    dict
        Dictionary mapping strategy names to their return data
    """
    strategies = {}
    
    for file_name, strategy_name, value_column in strategy_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Handle various data formats
                    if isinstance(data, pd.DataFrame):
                        # Check if the expected value column exists
                        if value_column in data.columns:
                            # Prepare the dataframe (set datetime index)
                            date_column = 'date'
                            if 'Date' in data.columns:
                                date_column = 'Date'
                            
                            data = prepare_dataframe(data, date_column)
                            
                            # Calculate daily returns
                            returns = calculate_daily_returns(data, value_column)
                            strategies[strategy_name] = returns
                        else:
                            print(f"Warning: '{value_column}' column not found in {file_name}")
                    else:
                        print(f"Warning: Data in {file_name} is not a DataFrame")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    return strategies

def prepare_data_for_anova(strategies):
    """
    Convert strategy returns into a format suitable for ANOVA analysis
    """
    data = []
    
    for strategy_name, returns in strategies.items():
        # Convert returns to DataFrame if it's a Series
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name='returns')
        
        # Add strategy column
        returns['strategy'] = strategy_name
        
        # Append to data list
        data.append(returns)
    
    # Concatenate all data
    combined_data = pd.concat(data, ignore_index=True)
    return combined_data

def perform_anova(data):
    """
    Perform one-way ANOVA test to determine if there are statistically 
    significant differences between strategy returns
    """
    # Create a model specification
    model = ols('returns ~ C(strategy)', data=data).fit()
    
    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Return F-statistic and p-value
    f_stat = anova_table.iloc[0]['F']
    p_val = anova_table.iloc[0]['PR(>F)']
    
    return anova_table, f_stat, p_val

def calculate_confidence_intervals(strategies):
    """
    Calculate 95% confidence intervals for each strategy's mean return
    """
    confidence_intervals = {}
    
    for strategy_name, returns in strategies.items():
        if isinstance(returns, pd.Series):
            returns = returns.values
        elif isinstance(returns, pd.DataFrame):
            returns = returns['returns'].values
        
        mean = np.mean(returns)
        std_err = stats.sem(returns)
        ci = stats.t.interval(0.95, len(returns)-1, loc=mean, scale=std_err)
        
        confidence_intervals[strategy_name] = {
            'mean': mean,
            'std_err': std_err,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'sample_size': len(returns),
            'std_dev': np.std(returns)
        }
    
    return confidence_intervals

def plot_confidence_intervals(confidence_intervals):
    """
    Plot means with 95% confidence intervals for each strategy
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort strategies by mean return for better visualization
    sorted_strategies = sorted(confidence_intervals.items(), key=lambda x: x[1]['mean'], reverse=True)
    strategies = [s[0] for s in sorted_strategies]
    means = [s[1]['mean'] for s in sorted_strategies]
    ci_lower = [s[1]['ci_lower'] for s in sorted_strategies]
    ci_upper = [s[1]['ci_upper'] for s in sorted_strategies]
    
    # Calculate error bar heights
    errors_minus = [means[i] - ci_lower[i] for i in range(len(means))]
    errors_plus = [ci_upper[i] - means[i] for i in range(len(means))]
    
    # Define color mapping to match the line chart
    color_map = {
        'PrimoRL': 'orange',
        'PrimoRL PPO': 'green',
        'PrimoRL Buy and Hold': 'darkgreen',
        'Buy and Hold': 'red',
        'FinRL': 'blue',
        'MACD': 'purple',
        'Momentum': 'brown',
        'Price MA': 'gray',
        'Mean-Variance': 'purple'
    }
    
    # Assign colors based on strategy names
    colors = [color_map.get(strategy, 'lightgray') for strategy in strategies]
    
    # Create bar chart with error bars
    positions = np.arange(len(strategies))
    bars = ax.bar(
        positions, 
        means, 
        yerr=[errors_minus, errors_plus], 
        capsize=4, 
        color=colors,
        alpha=0.7,
        width=0.6
    )
    
    # Format axes
    ax.set_xticks(positions)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_ylabel('Mean Daily Return')
    ax.set_title('Mean Returns with 95% Confidence Intervals by Strategy')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ypos = height + (0.0001 if height >= 0 else -0.0002)
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            ypos,
            f'{means[i]:.2%}',
            ha='center', 
            va='bottom' if height >= 0 else 'top',
            fontsize=12
        )
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'confidence_intervals_plot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def perform_pairwise_ttests(strategies):
    """
    Perform pairwise t-tests comparing PrimoRL with each other strategy
    """
    results = []
    
    if 'PrimoRL' not in strategies:
        print("PrimoRL strategy not found in the data.")
        return None
    
    # Get PrimoRL returns
    primorl_returns = strategies['PrimoRL']
    if isinstance(primorl_returns, pd.DataFrame):
        primorl_returns = primorl_returns['returns'].values
    elif isinstance(primorl_returns, pd.Series):
        primorl_returns = primorl_returns.values
    
    # Compare with each other strategy
    for strategy_name, returns in strategies.items():
        if strategy_name == 'PrimoRL':
            continue
        
        if isinstance(returns, pd.DataFrame):
            returns = returns['returns'].values
        elif isinstance(returns, pd.Series):
            returns = returns.values
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(primorl_returns, returns, equal_var=False)
        
        # Calculate mean difference
        mean_diff = np.mean(primorl_returns) - np.mean(returns)
        
        # Determine if PrimoRL is better
        is_better = "Yes" if mean_diff > 0 else "No"
        
        # Determine if difference is significant
        is_significant = "Yes" if p_val < 0.05 else "No"
        
        results.append({
            "PrimoRL vs": strategy_name,
            "Mean Difference": mean_diff,
            "% Difference": f"{(mean_diff / np.abs(np.mean(returns)) * 100):.2f}%" if np.mean(returns) != 0 else "N/A",
            "t-statistic": t_stat,
            "p-value": p_val,
            "PrimoRL Better?": is_better,
            "Statistically Significant?": is_significant
        })
    
    return pd.DataFrame(results)

def create_summary_table(confidence_intervals):
    """
    Create a summary table of all strategy statistics
    """
    summary_data = []
    
    for strategy_name, ci_data in confidence_intervals.items():
        summary_data.append({
            'Strategy': strategy_name,
            'Mean Return': ci_data['mean'],
            'Mean Return (%)': f"{ci_data['mean']:.4%}",
            'Std Dev': ci_data['std_dev'],
            'Std Dev (%)': f"{ci_data['std_dev']:.4%}",
            'CI Lower': ci_data['ci_lower'],
            'CI Upper': ci_data['ci_upper'],
            'CI Range (%)': f"[{ci_data['ci_lower']:.4%}, {ci_data['ci_upper']:.4%}]",
            'Sample Size': ci_data['sample_size']
        })
    
    # Convert to DataFrame and sort by mean return
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean Return', ascending=False)
    
    # Reorder columns for better presentation
    summary_df = summary_df[['Strategy', 'Mean Return (%)', 'Std Dev (%)', 
                            'CI Range (%)', 'Sample Size']]
    
    return summary_df

def save_results_to_markdown(results, output_dir=OUTPUT_DIR):
    """
    Save analysis results to a Markdown file
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    output_dir : str, optional
        Directory to save the markdown file, default is OUTPUT_DIR
    """
    file_path = os.path.join(output_dir, 'statistical_analysis_results.md')
    
    with open(file_path, 'w') as f:
        # Write title
        f.write("# Statistical Analysis of Trading Strategies\n\n")
        
        # ANOVA results
        f.write("## ANOVA Test Results\n\n")
        f.write("One-way ANOVA test was performed to determine if there are statistically ")
        f.write("significant differences between strategy returns.\n\n")
        f.write(f"**ANOVA p-value:** {results['p_value']:.6f}\n\n")
        
        if results['p_value'] < 0.05:
            f.write("The ANOVA test indicates that there are statistically significant differences ")
            f.write("between the performance of different strategies (p < 0.05).\n\n")
        else:
            f.write("The ANOVA test does not provide sufficient evidence of statistically significant ")
            f.write("differences between the performance of different strategies (p >= 0.05).\n\n")
        
        # ANOVA table
        f.write("### ANOVA Table\n\n")
        f.write("```\n")
        f.write(results['anova_results'].to_string())
        f.write("\n```\n\n")
        
        # Summary table
        f.write("## Strategy Performance Summary\n\n")
        f.write(results['summary_table'].to_markdown(index=False))
        f.write("\n\n")
        
        # Confidence intervals plot
        f.write("## Confidence Intervals\n\n")
        f.write('<img src="confidence_intervals_plot.png" alt="Mean Returns with 95% Confidence Intervals by Strategy" style="max-width: 1000px; width: 100%;" />\n\n')
        
        # Pairwise comparisons
        if results['pairwise_results'] is not None:
            f.write("## Pairwise Comparisons with PrimoRL\n\n")
            f.write(results['pairwise_results'].to_markdown(index=False))
            f.write("\n\n")
            
            # Statistical significance summary
            if 'PrimoRL' in results['confidence_intervals']:
                outperforms = sum(1 for row_idx, row in results['pairwise_results'].iterrows() 
                                if row["PrimoRL Better?"] == "Yes")
                sig_outperforms = sum(1 for row_idx, row in results['pairwise_results'].iterrows() 
                                    if row["PrimoRL Better?"] == "Yes" and row["Statistically Significant?"] == "Yes")
                
                f.write("## Statistical Significance Summary\n\n")
                f.write(f"PrimoRL outperforms {outperforms} out of {len(results['pairwise_results'])} other strategies\n\n")
                f.write(f"PrimoRL significantly outperforms {sig_outperforms} out of {len(results['pairwise_results'])} ")
                f.write("other strategies (p < 0.05)\n")
    
    print(f"Results saved to {file_path}")

def main():
    # Define strategy files
    strategy_files = [
        ('msft_primorl_df_account_value_ppo.pkl', 'PrimoRL', 'account_value'),
        ('msft_finrl_df_account_value_ppo.pkl', 'FinRL', 'account_value'),
        ('msft_finrl_buy_and_hold.pkl', 'Buy and Hold', 'close'),
        ('msft_macd_strategy.pkl', 'MACD', 'Value'),
        ('msft_momentum_strategy.pkl', 'Momentum', 'Value'),
        ('msft_p_ma_strategy.pkl', 'Price MA', 'Value')
    ]
    
    # Load strategy returns
    strategies = load_strategy_returns(strategy_files)
    if not strategies:
        print("Error: No strategy data loaded. Please check file paths.")
        return
    
    # Prepare data for analysis
    data = prepare_data_for_anova(strategies)
    
    # Perform ANOVA test
    anova_results, f_stat, p_val = perform_anova(data)
    print(f"ANOVA p-value: {p_val:.6f}")
    
    # Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(strategies)
    
    # Create summary table
    summary_table = create_summary_table(confidence_intervals)
    print(summary_table)
    
    # Perform pairwise t-tests
    pairwise_results = perform_pairwise_ttests(strategies)
    print("\n=== Pairwise Comparisons with PrimoRL ===")
    print(pairwise_results)
    
    # Create visualizations
    plot_confidence_intervals(confidence_intervals)
    
    # Summarize statistical significance 
    if 'PrimoRL' in confidence_intervals:
        outperforms = sum(1 for row_idx, row in pairwise_results.iterrows() 
                          if row["PrimoRL Better?"] == "Yes")
        sig_outperforms = sum(1 for row_idx, row in pairwise_results.iterrows() 
                             if row["PrimoRL Better?"] == "Yes" and row["Statistically Significant?"] == "Yes")
        
        print("\n=== Statistical Significance Summary ===")
        print(f"PrimoRL outperforms {outperforms} out of {len(pairwise_results)} other strategies")
        print(f"PrimoRL significantly outperforms {sig_outperforms} out of {len(pairwise_results)} other strategies (p < 0.05)")
    
    # Prepare results dictionary
    results = {
        'strategies': strategies,
        'anova_results': anova_results,
        'p_value': p_val,
        'confidence_intervals': confidence_intervals,
        'summary_table': summary_table,
        'pairwise_results': pairwise_results
    }
    
    # Save results to markdown file
    save_results_to_markdown(results)
    
    return results

if __name__ == "__main__":
    main()