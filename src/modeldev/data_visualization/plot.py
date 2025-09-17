import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import json
from src.modeldev.data_preprocessing.data_preprocessor import DataPreprocessor
import matplotlib.pyplot as plt

def create_auc_comparison_plots(file1_path, file2_path,
                                model1_name='CRIPD Model',
                                model2_name='Lightgbm Model',
                                save_path=None,
                                figsize=(12, 10),
                                show_stats=True,
                                separate_plots=False,
                                yearly=False):
    """
    Create comparison plots for AUC and PR AUC scores from two CSV files.

    Parameters:
    -----------
    file1_path : str
        Path to the first CSV file (e.g., 'cripd_mly_5years.csv')
    file2_path : str
        Path to the second CSV file (e.g., 'mthly_backtest_results_5yrs.csv')
    model1_name : str, default 'CRIPD Model'
        Name for the first model in legends
    model2_name : str, default 'Backtest Model'
        Name for the second model in legends
    save_path : str, optional
        Path to save the plot (e.g., 'auc_comparison.png')
    figsize : tuple, default (12, 10)
        Figure size (width, height)
    show_stats : bool, default True
        Whether to print summary statistics
    separate_plots : bool, default False
        If True, create separate figures for AUC and PR AUC

    Returns:
    --------
    merged_df : pd.DataFrame
        The merged dataframe with all data
    fig : matplotlib.figure.Figure or tuple
        The figure object(s)
    """

    # Read the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    df2 = df2.dropna(subset=['auc_score'])
    if yearly:
        df1['date'] = pd.to_datetime(df1['date'])
        df2['date'] = pd.to_datetime(df2['year'])
    # Convert date columns to datetime
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])

    # Merge the dataframes on date
    merged_df = pd.merge(df1, df2, on='date', suffixes=('_model1', '_model2'))

    if separate_plots:
        # Create separate figures
        fig1, ax1 = plt.subplots(figsize=figsize)
        fig2, ax2 = plt.subplots(figsize=figsize)

        # AUC Plot
        ax1.plot(merged_df['date'], merged_df['auc_score_model1'],
                 label=model1_name, color='#2563eb', linewidth=2, marker='o', markersize=4)
        ax1.plot(merged_df['date'], merged_df['auc_score_model2'],
                 label=model2_name, color='#dc2626', linewidth=2, marker='s', markersize=4)

        ax1.set_title('AUC Score Comparison', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('AUC Score', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(merged_df[['auc_score_model1', 'auc_score_model2']].min().min() - 0.01,
                     merged_df[['auc_score_model1', 'auc_score_model2']].max().max() + 0.01)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # PR AUC Plot
        ax2.plot(merged_df['date'], merged_df['pr_auc_model1'],
                 label=model1_name, color='#059669', linewidth=2, marker='o', markersize=4)
        ax2.plot(merged_df['date'], merged_df['pr_auc_model2'],
                 label=model2_name, color='#7c2d12', linewidth=2, marker='s', markersize=4)

        ax2.set_title('PR AUC Score Comparison', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('PR AUC Score', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(merged_df[['pr_auc_model1', 'pr_auc_model2']].min().min() - 0.01,
                     merged_df[['pr_auc_model1', 'pr_auc_model2']].max().max() + 0.01)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        fig1.tight_layout()
        fig2.tight_layout()

        if save_path:
            fig1.savefig(save_path.replace('.', '_auc.'), dpi=300, bbox_inches='tight')
            fig2.savefig(save_path.replace('.', '_pr_auc.'), dpi=300, bbox_inches='tight')

        plt.show()
        fig = (fig1, fig2)

    else:
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle('Model Performance Comparison Over Time', fontsize=16, fontweight='bold')

        # Plot 1: AUC Score Comparison
        ax1.plot(merged_df['date'], merged_df['auc_score_model1'],
                 label=model1_name, color='#2563eb', linewidth=2, marker='o', markersize=4)
        ax1.plot(merged_df['date'], merged_df['auc_score_model2'],
                 label=model2_name, color='#dc2626', linewidth=2, marker='s', markersize=4)

        ax1.set_title('AUC Score Comparison', fontsize=14, fontweight='semibold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('AUC Score', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(merged_df[['auc_score_model1', 'auc_score_model2']].min().min() - 0.01,
                     merged_df[['auc_score_model1', 'auc_score_model2']].max().max() + 0.01)

        # Format x-axis for dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: PR AUC Score Comparison
        ax2.plot(merged_df['date'], merged_df['pr_auc_model1'],
                 label=model1_name, color='#2563eb', linewidth=2, marker='o', markersize=4)
        ax2.plot(merged_df['date'], merged_df['pr_auc_model2'],
                 label=model2_name, color='#dc2626', linewidth=2, marker='s', markersize=4)

        ax2.set_title('PR AUC Score Comparison', fontsize=14, fontweight='semibold', pad=20)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('PR AUC Score', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(merged_df[['pr_auc_model1', 'pr_auc_model2']].min().min() - 0.01,
                     merged_df[['pr_auc_model1', 'pr_auc_model2']].max().max() + 0.01)

        # Format x-axis for dates
        ax3_twin = ax3.twinx()

        # Bar plot for test positives
        bars = ax3.bar(
            merged_df['date'], merged_df['test_positives'],
            label='Test Positives', color='#d62728', alpha=0.7
        )

        # Add number labels on each bar
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=8, color='#d62728'
            )

        # Optional twin axis for positive rate (line plot)
        # line2 = ax3_twin.plot(
        #     merged_df['date'], merged_df['positive_rate'],
        #     label='Positive Rate (%)', color='#9467bd',
        #     linewidth=2, marker='s', markersize=3
        # )

        ax3.set_title('Test Data: Positives Count', fontsize=12, fontweight='semibold')
        ax3.set_xlabel('Date', fontsize=10)
        ax3.set_ylabel('Test Positives Count', fontsize=10, color='#d62728')
        ax3_twin.set_ylabel('Positive Rate', fontsize=10, color='#9467bd')

        # Combine legends
        # lines = line1
        # labels = [l.get_label() for l in lines]
        # ax3.legend(lines, labels, loc='upper left', fontsize=9)

        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, fontsize=9)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show the plots
        plt.show()

    # Print summary statistics
    if show_stats:
        print("=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        print(f"\nAUC Scores:")
        print(f"{model1_name:15} - Mean: {merged_df['auc_score_model1'].mean():.4f}, "
              f"Std: {merged_df['auc_score_model1'].std():.4f}")
        print(f"{model2_name:15} - Mean: {merged_df['auc_score_model2'].mean():.4f}, "
              f"Std: {merged_df['auc_score_model2'].std():.4f}")

        print(f"\nPR AUC Scores:")
        print(f"{model1_name:15} - Mean: {merged_df['pr_auc_model1'].mean():.4f}, "
              f"Std: {merged_df['pr_auc_model1'].std():.4f}")
        print(f"{model2_name:15} - Mean: {merged_df['pr_auc_model2'].mean():.4f}, "
              f"Std: {merged_df['pr_auc_model2'].std():.4f}")

        print(f"\nCorrelation between models:")
        print(f"AUC Correlation: {merged_df['auc_score_model1'].corr(merged_df['auc_score_model2']):.4f}")
        print(f"PR AUC Correlation: {merged_df['pr_auc_model1'].corr(merged_df['pr_auc_model2']):.4f}")

    return merged_df, fig

# Example usage:
if __name__ == "__main__":
    # Basic usage
    # df, fig = create_auc_comparison_plots('cripd_mly_5years.csv',
    #                                       'mthly_backtest_results_5yrs.csv', save_path='cripd_vs_mlpd_mthly.png')
    df, fig = create_auc_comparison_plots('cripd_yearly_10years.csv',
                                          'yearly_backtest_results_10yrs.csv', save_path='cripd_vs_mlpd_yearly.png', yearly=True)
    # Advanced usage with custom parameters
    # df, fig = create_auc_comparison_plots(
    #     'cripd_mly_5years.csv',
    #     'mthly_backtest_results_5yrs.csv',
    #     model1_name='My Custom Model',
    #     model2_name='Baseline Model',
    #     save_path='model_comparison.png',
    #     figsize=(14, 12),
    #     separate_plots=True
    # )

# If you want separate plots instead of subplots, use this alternative:
"""
# Alternative: Create separate figures
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(merged_df['date'], merged_df['auc_score_cripd'], 
         label='CRIPD Model', color='#2563eb', linewidth=2, marker='o', markersize=4)
ax1.plot(merged_df['date'], merged_df['auc_score_backtest'], 
         label='Backtest Model', color='#dc2626', linewidth=2, marker='s', markersize=4)
ax1.set_title('AUC Score Comparison', fontsize=14, fontweight='semibold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('AUC Score', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(merged_df['date'], merged_df['pr_auc_cripd'], 
         label='CRIPD Model', color='#059669', linewidth=2, marker='o', markersize=4)
ax2.plot(merged_df['date'], merged_df['pr_auc_backtest'], 
         label='Backtest Model', color='#7c2d12', linewidth=2, marker='s', markersize=4)
ax2.set_title('PR AUC Score Comparison', fontsize=14, fontweight='semibold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('PR AUC Score', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.show()
"""