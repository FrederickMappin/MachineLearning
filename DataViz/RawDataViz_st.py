import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import missingno as msno
from scipy.stats import skew
import numpy as np

def calculate_statistics(df):
    # Initialize a list to store the statistics for each column
    stats_list = []
    processed_columns = []

    # Iterate through each column in the DataFrame
    for column in df.columns:
        try:
            col_data = df[column].dropna().astype(float)  # Drop NaN values and convert to float
            mean = col_data.mean()
            median = col_data.median()
            mode = col_data.mode().iloc[0] if not col_data.mode().empty else np.nan
            std_dev = col_data.std()
            min_val = col_data.min()
            max_val = col_data.max()
            skewness = skew(col_data)

            # Append the statistics to the list
            stats_list.append([mean, median, mode, std_dev, min_val, max_val, skewness])
            processed_columns.append(column)
        except ValueError:
            st.warning(f"Column '{column}' contains non-numerical values and will be skipped.")

    # Create a DataFrame from the statistics list
    stats_df = pd.DataFrame(stats_list, columns=['Mean', 'Median', 'Mode', 'Standard Deviation', 'Min', 'Max', 'Skewness'], index=processed_columns)

    return stats_df

def plot_histograms(df, columns_to_plot):
    # Filter the DataFrame to include only the specified columns
    df = df[columns_to_plot]

    # Determine the number of columns to plot
    num_cols = df.shape[1]
    cols = 4  # Number of columns for subplots
    rows = math.ceil(num_cols / cols)  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate through each column in the DataFrame
    for i, column in enumerate(df.columns):
        try:
            numeric_data = df[column].dropna().astype(float)
            axes[i].hist(numeric_data, bins=10)
            axes[i].set_title(f'Histogram for {column}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        except ValueError:
            st.warning(f"Column '{column}' contains non-numerical values and will be skipped.")
            fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)

def plot_boxplots(df, columns_to_plot):
    # Filter the DataFrame to include only the specified columns
    df = df[columns_to_plot]

    # Determine the number of columns to plot
    num_cols = df.shape[1]
    cols = 4  # Number of columns for subplots
    rows = math.ceil(num_cols / cols)  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate through each column in the DataFrame
    for i, column in enumerate(df.columns):
        try:
            numeric_data = df[column].dropna().astype(float)
            axes[i].boxplot(numeric_data, vert=False)
            axes[i].set_title(f'Box Plot for {column}')
            axes[i].set_xlabel('Value')
        except ValueError:
            st.warning(f"Column '{column}' contains non-numerical values and will be skipped.")
            fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)

def plot_barplots(df, columns_to_plot):
    # Filter the DataFrame to include only the specified columns
    df = df[columns_to_plot]

    # Determine the number of columns to plot
    num_cols = df.shape[1]
    cols = 4  # Number of columns for subplots
    rows = math.ceil(num_cols / cols)  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate through each column in the DataFrame
    for i, column in enumerate(df.columns):
        try:
            numeric_data = df[column].dropna().astype(float)
            axes[i].bar(range(len(numeric_data)), numeric_data)
            axes[i].set_title(f'Bar Plot for {column}')
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Value')
        except ValueError:
            st.warning(f"Column '{column}' contains non-numerical values and will be skipped.")
            fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)

def plot_missing_map(df):
    # Create the missing value map with black for present values and gray for missing values
    fig, ax = plt.subplots(figsize=(15, 5))
    msno.matrix(df, ax=ax, color=(0.5, 0.5, 0.5), sparkline=False)
    ax.patch.set_facecolor('gray')  # Set the background color to gray
    st.pyplot(fig)

def plot_correlation_matrix(df):
    # Drop columns with non-numerical values
    df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='any')

    # Calculate the correlation matrix
    correlations = df_numeric.corr()

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(df_numeric.columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df_numeric.columns, rotation=90)
    ax.set_yticklabels(df_numeric.columns)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

def plot_scatter_matrix(df):
    # Drop columns with non-numerical values
    df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='any')

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot the scatter plot matrix
    pd.plotting.scatter_matrix(df_numeric, ax=ax)
    plt.title('Scatter Plot Matrix')

    # Pass the figure to st.pyplot
    st.pyplot(fig)




# Streamlit app
st.title("Data Visualization and Statistics App")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    columns = df.columns.tolist()
    start_col = st.number_input("Start Column Index", min_value=0, max_value=len(columns)-1, value=0)
    end_col = st.number_input("End Column Index", min_value=0, max_value=len(columns)-1, value=len(columns)-1)

    columns_to_plot = columns[start_col:end_col+1]

    if st.button("Plot Histograms"):
        plot_histograms(df, columns_to_plot)

    if st.button("Plot Boxplots"):
        plot_boxplots(df, columns_to_plot)

    if st.button("Plot Barplots"):
        plot_barplots(df, columns_to_plot)

    if st.button("Plot Missing Map"):
        plot_missing_map(df)

    if st.button("Calculate Statistics"):
        stats_df = calculate_statistics(df)
        st.write("Statistics:")
        st.write(stats_df)

    if st.button("Plot Correlation Matrix"):
        plot_correlation_matrix(df)

    if st.button("Plot Scatter Matrix"):
        plot_scatter_matrix(df)