import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import calplot


st.set_page_config(layout="wide")

# Data
# Try daily profile
df = pd.read_csv("https://raw.githubusercontent.com/marsgr6/r-scripts/refs/heads/master/data/energie_resultats.csv")
df = df.drop(columns="Energie cumule (kWh)")
df = df.dropna()

df[df.columns[1:]] = df[df.columns[1:]].applymap(lambda x: max(x, 0))

# Replace all 0 values with Q2 (median) of each column
for col in df.select_dtypes(include=['number']).columns:  # Only process numeric columns
    median_value = df[col].median()
    df[col] = df[col].replace(0, median_value)
    # Remove outliers
    df[col] = np.where(df[col] > np.quantile(df[col], 0.9995), median_value, df[col])


# Convert 'Date' to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Streamlit App
st.title("Data series")

# Sidebar for user configuration
st.sidebar.header("Configuration")

# Date range selection
start_date = st.sidebar.date_input(
    "Select start date:",
    value=df["Date"].min(),
    min_value=df["Date"].min(),
    max_value=df["Date"].max(),
)
end_date = st.sidebar.date_input(
    "Select end date:",
    value=df["Date"].max(),
    min_value=df["Date"].min(),
    max_value=df["Date"].max(),
)

# Ensure valid date range
if start_date > end_date:
    st.error("End date must be after start date.")
else:
    # Filter data based on selected date range
    filtered_df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    # Dropdown for column selection
    column = st.sidebar.selectbox(
        "Select a column to plot:",
        df.columns[1:3],  # Exclude Date and Day columns
    )

    # Aggregation method selection (mean or sum)
    aggregation_method = st.sidebar.selectbox(
        "Select aggregation method:",
        ["Mean", "Sum"]
    )
    
    # Chart type selection
    chart_type = st.sidebar.selectbox("Select chart type:", ["Line Plot", "Bar Chart", "Heatmap", "Profiles"])

    # Daily summary selection
    resample_frequency = st.sidebar.selectbox("Select summary type:", ["None", "Hourly", 
        "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])

    frequency_map = {
            "Hourly": "H",
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "M",
            "Quarterly": "Q",
            "Yearly": "Y"
        }

    frequency_map_prev = {
            "Daily": "H",
            "Weekly": "D",
            "Monthly": "W",
            "Quarterly": "M",
            "Yearly": "Q"
        }

    if chart_type not in ["Heatmap", "Profiles"]:

        # Checkbox to toggle error bars
        if aggregation_method == "Mean":
            show_error_bars = st.sidebar.checkbox("Show error bars")
        else: 
            show_error_bars = False

        if filtered_df.empty:
            st.warning("No data available for the selected date range.")
        else:
            if resample_frequency != "None":
                filtered_df.set_index("Date", inplace=True)  # Set Date as the index for resampling
                # Resample data
                # resampled_mean = filtered_df[column].resample(frequency_map[resample_frequency]).mean()
                # resampled_std = filtered_df[column].resample(frequency_map[resample_frequency]).std()

                resampled_std = filtered_df[column].resample(frequency_map[resample_frequency])

                # Choose aggregation function
                if aggregation_method == "Mean":
                    if resample_frequency in ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]:
                        resampled_data = filtered_df[column].resample(frequency_map_prev[resample_frequency]).sum()
                        resampled_std = resampled_data.resample(frequency_map[resample_frequency]).std()
                        resampled_data = resampled_data.resample(frequency_map[resample_frequency]).mean()
                    else:
                        resampled_data = filtered_df[column].resample(frequency_map[resample_frequency]).sum()
                elif aggregation_method == "Sum":
                    resampled_data = filtered_df[column].resample(frequency_map[resample_frequency]).sum() 



                # Prepare the data for Plotly
                summary_df = pd.DataFrame({
                    "Date": resampled_data.index,
                    "Value": resampled_data.values,
                    "Std": resampled_std.values
                })

                # Prepare the plot
                if chart_type == "Line Plot":
                    if show_error_bars:
                        fig = px.line(
                            summary_df,
                            x="Date",
                            y="Value",
                            error_y="Std",
                            title=f"{chart_type} of {column} ({resample_frequency}) - {aggregation_method} with Error Bars"
                        )
                    else:
                        fig = px.line(
                            summary_df,
                            x="Date",
                            y="Value",
                            title=f"{chart_type} of {column} ({resample_frequency}) - {aggregation_method}"
                        )
                elif chart_type == "Bar Chart":
                    if show_error_bars:
                        fig = px.bar(
                            summary_df,
                            x="Date",
                            y="Value",
                            error_y="Std",
                            title=f"{chart_type} of {column} ({resample_frequency}) - {aggregation_method} with Error Bars"
                        )
                    else:
                        fig = px.bar(
                            summary_df,
                            x="Date",
                            y="Value",
                            title=f"{chart_type} of {column} ({resample_frequency}) - {aggregation_method}"
                        )


            else:
                # Plot raw data
                if chart_type == "Line Plot":
                    fig = px.line(filtered_df, x="Date", y=column, title=f"{chart_type} of {column}")
                elif chart_type == "Bar Chart":
                    fig = px.bar(filtered_df, x="Date", y=column, title=f"{chart_type} of {column}")

            # Show the plot
            st.plotly_chart(fig)

    if chart_type == "Profiles":
        ########################################################################################
        # PROFILES
        ########################################################################################

        # Streamlit app
        st.title("Daily Energy Profile")

        #filtered_df.set_index("Date", inplace=True)  # Set Date as the index for resampling

        # Extract day of the week and hour
        filtered_df["Day of Week"] = filtered_df["Date"].dt.day_name()  # Extract day name
        filtered_df["Hour"] = filtered_df["Date"].dt.hour  # Extract hour
        filtered_df["Year"] = filtered_df["Date"].dt.year  # Extract year

        # Group by Day of Week and Hour
        weekly_profile = filtered_df.groupby(["Day of Week", "Hour"])[column].mean().reset_index()
        if column == "Energie par periode (kWh)":
            weekly_profile = filtered_df.groupby(["Day of Week", "Hour"])[column].sum().reset_index()

        # Sort days in logical order
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_profile["Day of Week"] = pd.Categorical(weekly_profile["Day of Week"], categories=days_order, ordered=True)
        weekly_profile = weekly_profile.sort_values(["Day of Week", "Hour"]).reset_index(drop=True)


        # Dropdown for selecting day of the week
        #selected_day = st.selectbox("Select a day of the week", days_order)

        # Filter data for the selected day
        #filtered_data = weekly_profile[weekly_profile["Day of Week"] == selected_day]

        # Plot 24-hour profile for the selected day
        fig = px.line(
            weekly_profile,
            x="Hour",
            y=column,
            color="Day of Week",
            title=f"24-Hour Energy Profile",
            labels={"Hour": "Hour of the Day", column: column},
            markers=True,
        )

        # Display the plot
        st.plotly_chart(fig)


        # Extract month, day of month, and hour
        filtered_df["Month"] = filtered_df["Date"].dt.month_name()  # Extract month name
        filtered_df["Day of Month"] = filtered_df["Date"].dt.day  # Extract day of the month
        filtered_df["Hour"] = filtered_df["Date"].dt.hour  # Extract hour

        # Group by Month, Day of Month, and Hour
        monthly_profile = filtered_df.groupby(["Month", "Day of Month"])[column].mean().reset_index()

        if column == "Energie par periode (kWh)":
            monthly_profile = filtered_df.groupby(["Month", "Day of Month"])[column].sum().reset_index()



        # Sort months in logical order
        months_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        monthly_profile["Month"] = pd.Categorical(monthly_profile["Month"], categories=months_order, ordered=True)
        monthly_profile = monthly_profile.sort_values(["Month", "Day of Month"]).reset_index(drop=True)

        # Streamlit app
        st.title("Typical Monthly Energy Profile")

        # Dropdown for selecting a month
        #selected_month = st.selectbox("Select a month", months_order)

        # Filter data for the selected month
        #filtered_data = monthly_profile[monthly_profile["Month"] == selected_month]

        # Plot typical daily profile across all days in the month
        fig = px.line(
            monthly_profile,
            x="Day of Month",
            y=column,
            color="Month",
            title=f"Typical Month Energy Profile",
            labels={"Day of Month": "Day of Month", "Energie par periode (kWh)": "Energy (kWh)"},
            markers=True,
        ).update_traces(visible="legendonly", selector=lambda t: not t.name in ["January"])


        # Display the plot
        st.plotly_chart(fig)


        # Streamlit app
        st.title("Typical Weekly Energy Profile by Month")

        # Group by Month, Day of Month, and Hour
        w_profile = filtered_df.groupby(["Month", "Day of Week"])[column].mean().reset_index()
        if column == "Energie par periode (kWh)":
            w_profile = filtered_df.groupby(["Month", "Day of Week"])[column].sum().reset_index()

        w_profile["Day of Week"] = pd.Categorical(w_profile["Day of Week"], categories=days_order, ordered=True)
        w_profile = w_profile.sort_values(["Month", "Day of Week"]).reset_index(drop=True)

        # Plot 24-hour profile for the selected day
        fig = px.line(
            w_profile,
            x="Day of Week",
            y=column,
            color="Month",
            title=f"24-Hour Energy Profile",
            labels={"Hour": "Hour of the Day", column: column},
            markers=True,
        )

        # Display the plot
        st.plotly_chart(fig)


        # Streamlit app
        st.title("Typical Monthly Energy Profile by Year")


        # Group by Month, Day of Month, and Hour
        w_profile = filtered_df.groupby(["Month", "Year"])[column].mean().reset_index()

        if column == "Energie par periode (kWh)":
            w_profile = filtered_df.groupby(["Month", "Year"])[column].sum().reset_index()

        w_profile["Month"] = pd.Categorical(w_profile["Month"], categories=months_order, ordered=True)
        w_profile = w_profile.sort_values(["Month", "Year"]).reset_index(drop=True)

        # Plot 24-hour profile for the selected day
        fig = px.line(
            w_profile,
            x="Month",
            y=column,
            color="Year",
            title=f"24-Hour Energy Profile",
            labels={"Hour": "Hour of the Day", column: column},
            markers=True,
        )

        # Display the plot
        st.plotly_chart(fig)

        # Streamlit app
        st.title("Typical Weekly Energy Profile by Year")

        # Group by Month, Day of Month, and Hour
        w_profile = filtered_df.groupby(["Year", "Day of Week"])[column].mean().reset_index()

        if column == "Energie par periode (kWh)":
            w_profile = filtered_df.groupby(["Year", "Day of Week"])[column].sum().reset_index()

        w_profile["Day of Week"] = pd.Categorical(w_profile["Day of Week"], categories=days_order, ordered=True)
        w_profile = w_profile.sort_values(["Day of Week", "Year"]).reset_index(drop=True)


        # Plot 24-hour profile for the selected day
        fig = px.line(
            w_profile,
            x="Day of Week",
            y=column,
            color="Year",
            title=f"24-Hour Energy Profile",
            labels={"Hour": "Hour of the Day", column: column},
            markers=True,
        )

        # Display the plot
        st.plotly_chart(fig)

    if chart_type == "Heatmap":
        filtered_df.set_index("Date", inplace=True)  # Set Date as the index for resampling
        fig, ax = calplot.calplot(filtered_df.resample('D').sum()[column], cmap="coolwarm", colorbar=True)
        # Display in Streamlit
        st.pyplot(fig)


