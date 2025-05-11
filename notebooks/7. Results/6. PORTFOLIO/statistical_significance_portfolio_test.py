import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, List

# File paths
DATA_DIR = 'data'
OUTPUT_DIR = 'analysis_results'

# Set up basic matplotlib style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def create_output_dir():
    """Create output directory if it doesn't exist"""
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

def load_strategy_returns(strategy_files, data_dir=DATA_DIR):
    """
    Load strategy returns from pickle files based on provided file list
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
                            print(f"Available columns: {data.columns.tolist()}")
                    else:
                        print(f"Warning: Data in {file_name} is not a DataFrame")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        else:
            print(f"Warning: File {file_name} not found in {data_dir}")
    
    return strategies

def calculate_confidence_intervals(returns_data: Dict[str, pd.Series], confidence_level: float = 0.95) -> pd.DataFrame:
    """Calculate confidence intervals for the mean daily return of each strategy"""
    ci_results = []
    
    for strategy, returns in returns_data.items():
        if isinstance(returns, pd.Series):
            returns_values = returns.values
        else:
            print(f"Warning: Strategy {strategy} returns is not a Series")
            continue
            
        n = len(returns_values)
        mean = np.mean(returns_values)
        se = stats.sem(returns_values)  # Standard error of the mean
        
        # Calculate confidence interval using t-distribution
        ci_lower, ci_upper = stats.t.interval(
            confidence_level,
            df=n-1,
            loc=mean,
            scale=se
        )
        
        ci_results.append({
            'Strategy': strategy,
            'Mean Daily Return': mean,
            'Daily CI Lower': ci_lower,
            'Daily CI Upper': ci_upper,
            'Sample Size': n
        })
    
    ci_df = pd.DataFrame(ci_results).set_index('Strategy')
    
    # Sort by Mean Daily Return in descending order
    ci_df = ci_df.sort_values('Mean Daily Return', ascending=False)
    
    return ci_df

def run_statistical_tests(returns_data: Dict[str, pd.Series]) -> Tuple[float, pd.DataFrame]:
    """Run statistical tests to compare strategies"""
    # Extract returns as lists for each strategy
    strategy_returns = {strategy: returns.values for strategy, returns in returns_data.items()}
    
    if len(strategy_returns) < 2:
        print("Not enough strategies to run ANOVA")
        return float('nan'), pd.DataFrame()
    
    # Run ANOVA test on all strategies
    anova_data = [returns for returns in strategy_returns.values()]
    f_stat, anova_p = stats.f_oneway(*anova_data)
    
    # Find PrimoRL strategies
    primorl_strategies = [s for s in strategy_returns.keys() if 'PrimoRL' in s]
    
    # If no PrimoRL strategies found, use the first one as reference
    if not primorl_strategies:
        print("No PrimoRL strategies found. Using the first strategy as reference.")
        reference_strategy = list(strategy_returns.keys())[0]
    else:
        # Use the first PrimoRL strategy as reference
        reference_strategy = primorl_strategies[0]
    
    # Run pairwise t-tests comparing the reference strategy to others
    t_test_results = []
    
    for strategy, returns in strategy_returns.items():
        if strategy != reference_strategy:
            t_stat, p_value = stats.ttest_ind(
                strategy_returns[reference_strategy],
                returns,
                equal_var=False  # Use Welch's t-test for unequal variances
            )
            
            t_test_results.append({
                'Reference': reference_strategy,
                'Compared To': strategy,
                't-statistic': t_stat,
                'p-value': p_value,
                'Significant': p_value < 0.05
            })
    
    # Convert to DataFrame
    t_test_df = pd.DataFrame(t_test_results)
    
    return anova_p, t_test_df

def create_visualizations(
    returns_data: Dict[str, pd.Series],
    ci_df: pd.DataFrame,
    anova_p: float,
    t_test_df: pd.DataFrame
):
    """Create confidence intervals and p-value visualizations using matplotlib"""
    create_output_dir()
    
    # 1. Confidence Intervals Visualization for Daily Returns
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group strategies by type
    portfolio_strategies = [s for s in ci_df.index if not (s.endswith('DJI') or s.endswith('Mean-Variance'))]
    dji_strategies = [s for s in ci_df.index if s.endswith('DJI')]
    mean_var_strategies = [s for s in ci_df.index if s.endswith('Mean-Variance')]
    
    # Combine all strategies in desired order
    all_strategies = portfolio_strategies + dji_strategies + mean_var_strategies
    
    # Get data for visualization
    strategies = [s for s in all_strategies if s in ci_df.index]
    
    means = [ci_df.loc[s, 'Mean Daily Return'] for s in strategies]
    ci_lower = [ci_df.loc[s, 'Daily CI Lower'] for s in strategies]
    ci_upper = [ci_df.loc[s, 'Daily CI Upper'] for s in strategies]
    
    # Calculate error bars
    errors_minus = [means[i] - ci_lower[i] for i in range(len(means))]
    errors_plus = [ci_upper[i] - means[i] for i in range(len(means))]
    
    # Define color mapping to match the line chart
    color_map = {
        'PrimoRL SAC': 'orange',
        'PrimoRL PPO': 'green',
        'DJI': 'red',
        'Mean-Variance': 'purple',
        'FinRL PPO': 'blue',
        'FinRL SAC': 'cornflowerblue',
    }
    
    # Assign colors based on strategy names
    colors = [color_map.get(strategy, 'gray') for strategy in strategies]
    
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
    ax.set_title('95% Confidence Intervals for Mean Daily Returns')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ypos = height + (0.0005 if height >= 0 else -0.0010)
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            ypos,
            f'{means[i]:.4%}',
            ha='center', 
            va='bottom' if height >= 0 else 'top',
            fontsize=10
        )
    
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/confidence_intervals_daily.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Statistical Significance Heatmap (p-value plot)
    if not t_test_df.empty and len(returns_data) >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a matrix for the heatmap
        strategies = list(returns_data.keys())
        
        if len(strategies) >= 2:  # Need at least 2 strategies for a meaningful heatmap
            p_value_matrix = pd.DataFrame(1.0, index=strategies, columns=strategies)
            
            # Fill in p-values from t-tests
            for _, row in t_test_df.iterrows():
                ref = row['Reference']
                comp = row['Compared To']
                
                p_value_matrix.loc[ref, comp] = row['p-value']
                p_value_matrix.loc[comp, ref] = row['p-value']
            
            # Create our own heatmap using matplotlib
            masked_array = np.ma.masked_array(
                p_value_matrix.values, 
                mask=np.triu(np.ones_like(p_value_matrix.values, dtype=bool))
            )
            
            # Create a custom colormap from blue to red (reversed coolwarm)
            cmap = plt.cm.coolwarm_r
            
            # Create the heatmap
            im = ax.imshow(masked_array, cmap=cmap, vmin=0, vmax=0.1)
            
            # Add color bar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('p-value')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(strategies)))
            ax.set_yticks(np.arange(len(strategies)))
            ax.set_xticklabels(strategies, rotation=45, ha='right')
            ax.set_yticklabels(strategies)
            
            # Add text annotations
            for i in range(len(strategies)):
                for j in range(len(strategies)):
                    if i > j:  # Only lower triangle
                        text = ax.text(
                            j, i, f"{p_value_matrix.iloc[i, j]:.4f}",
                            ha="center", va="center",
                            color="black" if p_value_matrix.iloc[i, j] > 0.05 else "white"
                        )
            
            ax.set_title(f'Pairwise T-Test p-values (ANOVA p={anova_p:.4f})')
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/statistical_significance_heatmap.png', dpi=300, bbox_inches='tight')
        else:
            print("Not enough strategies for heatmap visualization")
        
        plt.close(fig)

def run_analysis():
    """Main function to run the simplified analysis"""
    print("Loading data and calculating returns...")
    
    # Define strategy files with explicit names
    portfolio_files = [
        ('primorl_df_account_value_ppo.pkl', 'PrimoRL PPO', 'account_value'),
        ('primorl_df_account_value_sac.pkl', 'PrimoRL SAC', 'account_value'),
        ('finrl_df_account_value_ppo.pkl', 'FinRL PPO', 'account_value'),
        ('finrl_df_account_value_sac.pkl', 'FinRL SAC', 'account_value'),
    ]
    
    dji_files = [
        ('primorl_dji.pkl', 'DJI', 'close')
    ]
    
    mean_var_files = [
        ('primorl_mean_var.pkl', 'Mean-Variance', 'Mean Var')
    ]
    
    # Load all strategy returns
    print("Loading portfolio strategies...")
    portfolio_returns = load_strategy_returns(portfolio_files)
    
    print("Loading DJI strategies...")
    dji_returns = load_strategy_returns(dji_files)
    
    print("Loading mean-variance strategies...")
    mean_var_returns = load_strategy_returns(mean_var_files)
    
    # Combine all returns data
    all_returns = {}
    all_returns.update(portfolio_returns)
    all_returns.update(dji_returns)
    all_returns.update(mean_var_returns)
    
    print(f"Loaded {len(all_returns)} strategies in total")
    
    print("Calculating confidence intervals...")
    ci_df = calculate_confidence_intervals(all_returns)
    
    print("Running statistical tests...")
    # Run tests on ALL strategies
    anova_p, t_test_df = run_statistical_tests(all_returns)
    
    print("Creating visualizations...")
    create_visualizations(all_returns, ci_df, anova_p, t_test_df)
    
    print("\nResults:")
    print("\nConfidence Intervals (95%):")
    print(ci_df[['Mean Daily Return', 'Daily CI Lower', 'Daily CI Upper']].round(4))
    
    print(f"\nANOVA p-value: {anova_p:.6f}")
    print(f"Statistically significant differences between strategies: {anova_p < 0.05}")
    
    print(f"\nAll results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()