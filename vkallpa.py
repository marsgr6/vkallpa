import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import calplot
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore


def plot_energy_profile(df, groupby_cols, title, x_label):
        profile = df.groupby(groupby_cols)[column].mean().reset_index()
        fig = px.line(profile, x=groupby_cols[-1], y=column, color=groupby_cols[0], title=title, labels={groupby_cols[-1]: x_label, column: "Energy (kWh)"}, markers=True)
        st.plotly_chart(fig)

# Anomaly detection function using Z-score
def detect_anomalies_zscore(df, threshold=3):
    z_scores = zscore(df['Value'])
    anomalies = np.where(np.abs(z_scores) > threshold)[0]
    return anomalies

# Anomaly detection function using Isolation Forest
def detect_anomalies_iforest(df, contamination=0.1):
    model = IsolationForest(contamination=contamination)
    anomalies = model.fit_predict(df[['Value']])
    return np.where(anomalies == -1)[0]

st.set_page_config(layout="wide")

# Load and clean data
@st.cache_data
def load_and_clean_data():
    """Load and preprocess the dataset with caching for performance."""
    df = pd.read_csv("https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/data/energie_resultats.csv")
    df = df.drop(columns="Energie cumule (kWh)").dropna()
    df[df.columns[1:]] = df[df.columns[1:]].applymap(lambda x: max(x, 0))
    
    for col in df.select_dtypes(include=['number']).columns:
        median_value = df[col].median()
        df[col] = np.where(df[col] == 0, median_value, df[col])
        df[col] = np.where(df[col] > np.quantile(df[col], 0.9995), median_value, df[col])
    
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_and_clean_data()

# Sidebar filters
st.sidebar.header("Configuration")
start_date = st.sidebar.date_input("Select start date:", df["Date"].min(), df["Date"].min(), df["Date"].max())
end_date = st.sidebar.date_input("Select end date:", df["Date"].max(), df["Date"].min(), df["Date"].max())

if start_date > end_date:
    st.error("End date must be after start date.")
else:
    filtered_df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    column = st.sidebar.selectbox("Select a column to plot:", df.columns[1:3])
    aggregation_method = st.sidebar.selectbox("Select aggregation method:", ["Sum", "Mean"])
    chart_type = st.sidebar.selectbox("Select chart type:", ["Line Plot", "Bar Chart", "Heatmap", "Profiles", "Anomaly"])
    resample_frequency = st.sidebar.selectbox("Select summary type:", ["None", "Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
    show_error_bars = aggregation_method == "Mean" and st.sidebar.checkbox("Show error bars") and resample_frequency != "None"
    
    frequency_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
    frequency_map_prev = {"Daily": "H", "Weekly": "D", "Monthly": "W", "Quarterly": "M", "Yearly": "Q"}

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
    else:
        filtered_df.set_index("Date", inplace=True)
        if resample_frequency != "None":
            if aggregation_method == "Mean":
                resampled_data = filtered_df[column].resample(frequency_map_prev.get(resample_frequency, frequency_map[resample_frequency])).sum()
                resampled_std = resampled_data.resample(frequency_map[resample_frequency]).std()
                resampled_data = resampled_data.resample(frequency_map[resample_frequency]).mean()
            else:
                resampled_data = filtered_df[column].resample(frequency_map[resample_frequency]).sum()
                resampled_std = None
            summary_df = pd.DataFrame({"Date": resampled_data.index, "Value": resampled_data.values, "Std": resampled_std.values if resampled_std is not None else None})
        else:
            summary_df = filtered_df.reset_index()[["Date", column]].rename(columns={column: "Value"})
        
        if chart_type == "Line Plot":
            fig = px.line(summary_df, x="Date", y="Value", error_y="Std" if show_error_bars else None, title=f"{chart_type} of {column} ({resample_frequency}) - {aggregation_method}")
            st.plotly_chart(fig)
        if chart_type == "Bar Chart":
            fig = px.bar(summary_df, x="Date", y="Value", error_y="Std" if show_error_bars else None, title=f"{chart_type} of {column} ({resample_frequency}) - {aggregation_method}")
            st.plotly_chart(fig)
    
        if chart_type == "Profiles":
            DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            MONTHS_ORDER = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            
            filtered_df["Day of Week"] = pd.Categorical(filtered_df.index.day_name(), categories=DAYS_ORDER, ordered=True)
            filtered_df["Month"] = pd.Categorical(filtered_df.index.month_name(), categories=MONTHS_ORDER, ordered=True)
            filtered_df["Hour"] = filtered_df.index.hour
            filtered_df["Year"] = filtered_df.index.year
            filtered_df["Day of Month"] = filtered_df.index.day
            
            plot_energy_profile(filtered_df, ["Day of Week", "Hour"], "24-Hour Energy Profile", "Hour of the Day")
            plot_energy_profile(filtered_df, ["Month", "Day of Month"], "Typical Monthly Energy Profile", "Day of Month")
            plot_energy_profile(filtered_df, ["Month", "Day of Week"], "Typical Weekly Energy Profile by Month", "Day of Week")
            plot_energy_profile(filtered_df, ["Year", "Month"], "Typical Monthly Energy Profile by Year", "Month")
            plot_energy_profile(filtered_df, ["Year", "Day of Week"], "Typical Weekly Energy Profile by Year", "Day of Week")

        if chart_type == "Heatmap":
            fig, ax = calplot.calplot(filtered_df.resample('D').sum()[column], cmap="coolwarm", colorbar=True)
            st.pyplot(fig)
        
        if chart_type == "Anomaly":
            # Additional dropdown to select anomaly detection method
            anomaly_method = st.sidebar.selectbox("Select anomaly detection method:", ["Isolation Forest", "Z-score"])
            
            if anomaly_method == "Z-score":
                zscore_threshold = st.sidebar.selectbox("Select Z-score threshold:", [1, 2, 3])
                anomalies = detect_anomalies_zscore(summary_df, threshold=zscore_threshold)  # Use Z-score for anomaly detection
            elif anomaly_method == "Isolation Forest":
                contamination_value = st.sidebar.selectbox("Select contamination parameter:", [0.05, 0.1, 0.2, 0.3])
                anomalies = detect_anomalies_iforest(summary_df, contamination=contamination_value)  # Use Isolation Forest for anomaly detection

            # Plot the time series with anomalies
            fig = px.line(summary_df, x="Date", y="Value", title=f"Anomaly Detection on {column} - {anomaly_method}")
            fig.add_scatter(x=summary_df["Date"].iloc[anomalies], y=summary_df["Value"].iloc[anomalies], mode='markers', marker=dict(color='red', size=10), name='Anomalies')
            st.plotly_chart(fig)
