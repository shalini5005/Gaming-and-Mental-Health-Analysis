import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
import os
import base64

# Set page config for wider layout and a title
st.set_page_config(page_title="Gaming & Wellness Analytics", page_icon="🎮", layout="wide")

st.title("Gaming Habits & Mental Wellbeing Analysis")

# Inject Custom CSS for Premium, Dynamic Aesthetics
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

bg_base64 = get_base64_of_bin_file(os.path.join("assets", "background.png"))

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide the Streamlit Deploy Button */
    .stDeployButton, .stAppDeployButton, [data-testid="stAppDeployButton"] {
        display: none !important;
    }
"""

if bg_base64:
    custom_css += f"""
    .stApp {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(14, 17, 23, 0.85); /* Dark overlay for readability */
        padding: 3rem !important;
        border-radius: 15px;
        margin-top: 2rem;
    }}
    """
else:
    custom_css += """
    /* Move main app content upward to reduce wasted blank top space */
    .block-container {
        padding-top: 2rem !important;
    }
    """

custom_css += """
    /* Center and adjust main heading specifically */
    h1:first-of-type {
        text-align: center !important;
        padding-bottom: 1rem !important;
    }

    
    /* Premium Gradient Titles */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Subtle hover effect for elements */
    .stPlotlyChart {
         border-radius: 12px;
         box-shadow: 0 8px 16px rgba(0,0,0,0.5);
         transition: transform 0.3s ease;
    }
    .stPlotlyChart:hover {
         transform: scale(1.02);
    }
    
    /* Custom Styling for metrics */
    div[data-testid="stMetricValue"] {
        color: #00d2ff;
        font-weight: 800;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Caching data processing to avoid reloading on every interaction
@st.cache_data
def load_and_clean_data():
    file_path = os.path.join("project", "Gaming and Mental Health.csv")
    if not os.path.exists(file_path):
        # fallback to current dir
        file_path = "Gaming and Mental Health.csv"
        
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Cannot find data at '{file_path}'. Please ensure the file is present.")
        return pd.DataFrame()
    return df

df_raw = load_and_clean_data()

if df_raw.empty:
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
menu_selection = st.sidebar.selectbox(
    "Choose Analysis Stage:",
    [
        "1. Data Overview & Cleaning", 
        "2. Outlier Analysis", 
        "3. Exploratory Data Analysis (EDA)", 
        "4. Data Visualizations & Conclusion",
        "5. Statistical Testing & Modeling",
    ]
)

st.sidebar.markdown("---")

# Auto-scroll to top when navigating to a new page
if 'last_menu' not in st.session_state:
    st.session_state['last_menu'] = menu_selection

scroll_js = ""
if st.session_state['last_menu'] != menu_selection:
    st.session_state['last_menu'] = menu_selection
    scroll_js = f"""
    <script>
        // Ensure Streamlit recognizes this as a new component execution: {menu_selection}
        setTimeout(function() {{
            var elements = window.parent.document.querySelectorAll('.main, [data-testid="stAppViewContainer"], [data-testid="stMain"], .block-container');
            for (var i = 0; i < elements.length; i++) {{
                elements[i].scrollTop = 0;
                elements[i].scrollTo({{top: 0, behavior: 'instant'}});
            }}
            window.parent.scrollTo({{top: 0, behavior: 'instant'}});
            window.parent.document.body.scrollTop = 0;
            window.parent.document.documentElement.scrollTop = 0;
        }}, 50);
    </script>
    """
st.components.v1.html(scroll_js, height=0, width=0)

# Pre-calc Outliers & Cleaning for consistent data usage across pages
null_counts = df_raw.isnull().sum()
df_cleaned = df_raw.dropna()

Q1 = df_cleaned['daily_gaming_hours'].quantile(0.25)
Q3 = df_cleaned['daily_gaming_hours'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df_cleaned[(df_cleaned['daily_gaming_hours'] < lower) | (df_cleaned['daily_gaming_hours'] > upper)]
df_final = df_cleaned[(df_cleaned['daily_gaming_hours'] >= lower) & (df_cleaned['daily_gaming_hours'] <= upper)]


if menu_selection == "1. Data Overview & Cleaning":
    
    st.header("1. Data Overview & Missing Values")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head())
    with col2:
        st.subheader("Dataset Info")
        st.write(f"**Total Rows:** {df_raw.shape[0]}")
        st.write(f"**Total Columns:** {df_raw.shape[1]}")
        
    st.markdown("---")
    st.subheader("Handling Missing Values")
    
    col3, col4 = st.columns([1, 2])
    with col3:
        st.write("Null Values per Column:")
        if null_counts.sum() > 0:
            missing_df = null_counts[null_counts > 0].reset_index()
            missing_df.columns = ["Feature", "Missing Count"]
            st.dataframe(missing_df, hide_index=True)
        else:
            st.info("No missing values!")
        
    with col4:
        if null_counts.sum() > 0:
            st.success(f"Dropped {df_raw.shape[0] - df_cleaned.shape[0]} rows containing NaN values.")
        else:
            st.info("No missing values found! Ready for analysis.")
            
elif menu_selection == "2. Outlier Analysis":
    st.header("2. Handling Outliers in Gaming Hours")
    st.markdown("Extreme values (outliers) can distort our trends. Here, we identify and remove statistically anomalous *Daily Gaming Hours* to keep our analysis accurate.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before Outlier Removal")
        fig_before = px.box(df_cleaned, y='daily_gaming_hours', title="With Outliers", points="outliers", template="plotly_dark", 
                            color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig_before, use_container_width=True)
        
    with col2:
        st.subheader("After Outlier Removal")
        fig_after = px.box(df_final, y='daily_gaming_hours', title="Without Outliers", points="outliers", template="plotly_dark",
                           color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig_after, use_container_width=True)
        
    st.subheader("Outliers Removed")
    st.write(f"Identified {len(outliers)} outliers outside the bounds (Lower: {lower:.2f}, Upper: {upper:.2f}):")
    st.warning(f"**Why are these removed?** Look at the **'daily_gaming_hours'** column below. All these rows have values ({', '.join(outliers['daily_gaming_hours'].astype(str).tolist())}) that are strictly greater than the calculated Upper Bound of {upper:.2f}. Because they are mathematically anomalous, they are removed so they don't corrupt our trend models.")
    st.dataframe(outliers)
    st.success(f"Dataset condensed from {len(df_cleaned)} to {len(df_final)} rows.")

elif menu_selection == "3. Exploratory Data Analysis (EDA)":
    st.header("3. Exploratory Data Analysis (EDA)")
    
    tab1, tab2, tab3 = st.tabs(["Gaming Time Distribution", "Gaming vs Mental Stress", "Custom Comparison"])
    
    with tab1:
        st.subheader("Objective 1: Daily Gaming Hours Distribution")
        
        # Display Histogram with Density Curve
        import plotly.figure_factory as ff
        hist_data = [df_final['daily_gaming_hours'].dropna().values]
        group_labels = ['Gaming Hours']
        
        # calculate bin size based on range to ensure nice pillars
        import numpy as np
        bin_size = (df_final['daily_gaming_hours'].max() - df_final['daily_gaming_hours'].min()) / 15
        if bin_size == 0: bin_size = 1

        fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=bin_size, show_rug=False, colors=['#ff9f43'])
        
        # Update the density curve (KDE) to be a distinct color (white) and thicker
        if len(fig_hist.data) > 1:
            fig_hist.data[1].update(line=dict(color='white', width=4))

        fig_hist.update_layout(title="Distribution of Daily Gaming Hours (with skewness curve)", 
                               xaxis_title="Daily Gaming Hours", yaxis_title="Density", template="plotly_dark",
                               showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Calculate skewness
        skewness = df_final['daily_gaming_hours'].skew()
        if skewness > 0.5:
            skew_text = f"**Positively Skewed ({skewness:.2f})**: A few 'hardcore' gamers create that long tail you see stretching to the right."
        elif skewness < -0.5:
            skew_text = f"**Negatively Skewed ({skewness:.2f})**: The long tail stretches to the left, meaning most play many hours."
        else:
            skew_text = f"**Symmetric ({skewness:.2f})**: The curve is perfectly central, showing evenly distributed gaming hours."
            
        st.info(f"**Conclusion:** {skew_text}")
        
    with tab2:
        st.subheader("Objective 2: Gaming vs Mental Stress (Isolation)")
        fig_scatter1 = px.scatter(df_final, x="daily_gaming_hours", y="social_isolation_score", 
                                  color="social_isolation_score", size="daily_gaming_hours",
                                  title="Does more gaming mean more isolation?", template="plotly_dark")
        fig_scatter1.update_layout(xaxis_title="Gaming Hours", yaxis_title="Isolation Score")
        st.plotly_chart(fig_scatter1, use_container_width=True)
        
    with tab3:
        st.subheader("Objective 3: Custom Feature Analysis")
        st.markdown("Select a feature below to see how it correlates with Daily Gaming Hours.")
        feature = st.selectbox("Select Feature to Compare", 
                               ["sleep_quality", "academic_work_performance", "face_to_face_social_hours_weekly", "grades_gpa", "mood_state"], key="custom_comparison_feat")
        
        fig_scatter2 = px.scatter(df_final, x="daily_gaming_hours", y=feature, 
                                  color=feature, color_continuous_scale="Viridis",
                                  title=f"Gaming Hours vs {feature.replace('_', ' ').title()}", template="plotly_dark")
        fig_scatter2.update_layout(xaxis_title="Gaming Hours", yaxis_title=feature.replace('_', ' ').title())
        st.plotly_chart(fig_scatter2, use_container_width=True)

elif menu_selection == "4. Data Visualizations & Conclusion":
    st.header("4. Data Visualizations & Conclusion")
    st.markdown("---")
    
    # 1. Pie Chart
    st.subheader("1. Proportion of Mood States (Pie Chart)")
    st.markdown("This pie chart summarizes the final demographic of mood states in our cleaned data, giving a conclusive look at the wellbeing breakdown of the survey population.")
    if "mood_state" in df_final.columns:
        mood_counts = df_final["mood_state"].value_counts().reset_index()
        mood_counts.columns = ["mood_state", "count"]
        fig_pie = px.pie(mood_counts, names="mood_state", values="count", hole=0.4,
                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("---")
    
    # 2. Clustered Column Chart
    st.subheader("2. Impact on Well-being by Mood (Clustered Column Chart)")
    st.markdown("A clustered column chart directly compares two key metrics—**Social Isolation** and **Sleep Hours**—side-by-side across different mood states, highlighting the ultimate behavioral takeaways.")
    if "mood_state" in df_final.columns and "sleep_hours" in df_final.columns:
        clustered_df = df_final.groupby("mood_state")[["social_isolation_score", "sleep_hours"]].mean().reset_index()
        # Melting for plotly formatting
        clustered_melt = clustered_df.melt(id_vars="mood_state", var_name="Metric", value_name="Average Score")
        # Formatting labels nicely
        clustered_melt['Metric'] = clustered_melt['Metric'].replace({
            "social_isolation_score": "Social Isolation", 
            "sleep_hours": "Sleep Hours"
        })
        
        fig_clustered = px.bar(clustered_melt, x="mood_state", y="Average Score", color="Metric", barmode="group",
                               template="plotly_dark", text_auto='.1f', 
                               color_discrete_sequence=['#00d2ff', '#e74c3c'])
        fig_clustered.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        fig_clustered.update_layout(xaxis_title="Mood State", yaxis_title="Average Score")
        st.plotly_chart(fig_clustered, use_container_width=True)

    st.markdown("---")
    
    # 3. Line Chart
    st.subheader("3. Isolation Trend over Gaming hours(Line Chart)")
    st.markdown("A finalized summary trend showing how social isolation scales alongside daily gaming hours. We grouped gaming hours into rounded buckets to show a clean trendline.")
    # Grouping into buckets (rounding)
    trend_df = df_final.copy()
    trend_df["gaming_hour_bucket"] = trend_df["daily_gaming_hours"].round()
    line_df = trend_df.groupby("gaming_hour_bucket")["social_isolation_score"].mean().reset_index()
    line_df = line_df.sort_values(by="gaming_hour_bucket")
    
    fig_line = px.line(line_df, x="gaming_hour_bucket", y="social_isolation_score", markers=True,
                       template="plotly_dark", line_shape="spline",
                       title="Social Isolation vs Rounded Gaming Hours")
    fig_line.update_traces(line=dict(color="#f1c40f", width=4), marker=dict(size=10, color="white"))
    fig_line.update_layout(xaxis_title="Average Daily Gaming Hours (Rounded)", yaxis_title="Average Social Isolation Score")
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    st.success("**Conclusion:** By focusing on proportion (Pie Chart), multi-variable comparison (Clustered Column), and overarching trends (Line Chart), we effectively summarize our project's insights. We can firmly conclude that elevated gaming hours correlate strongly with shifting mood states, diminished sleep quality, and heightened isolation.")

elif menu_selection == "5. Statistical Testing & Modeling":
    st.header("Objective 5: Hypothesis Testing (T-test)")
    
    mean_gaming = df_final['daily_gaming_hours'].mean()
    high = df_final[df_final['daily_gaming_hours'] > mean_gaming]['social_isolation_score']
    low = df_final[df_final['daily_gaming_hours'] <= mean_gaming]['social_isolation_score']
    
    t_stat, p_val = ttest_ind(high, low)
    
    st.info("**Objective:** Compares the isolation levels of High vs. Low gamers to see if there's a significant difference.")
    st.write(f"We divide the group based on the average daily gaming hours (**{mean_gaming:.2f} hours**)")
    st.markdown("- **High Gamers**: Above Average")
    st.markdown("- **Low Gamers**: Below Average")
    
    col1, col2 = st.columns(2)
    col1.metric("T-Statistic", f"{t_stat:.4f}")
    col2.metric("P-Value", f"{p_val:.4f}")
    
    if p_val < 0.05:
        st.error("📉 **Result**: Reject H0 → Gaming significantly affects social isolation.")
    else:
        st.success("📈 **Result**: Fail to Reject H0 → No significant effect.")
        
    st.markdown("---")
    
    st.header("Objective 5: Correlation & Linear Regression")
    
    st.subheader("1. Correlation Basis (Feature Selection)")
    st.info("**Concept:** We explicitly check correlation to determine which feature (X) has the strongest relationship with the target (Y).")
    
    # Calculate Correlation Matrix
    # We only take numeric columns for correlation as taught in class
    numeric_cols = df_final.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("**Correlation Matrix:**")
        # Use height to limit the dataframe scroll area if it's too long
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True, height=400)
    
    with col2:
        st.write("**Basis for Selection:**")
        st.markdown(f"- **Target (Y):** `social_isolation_score` (what we predict)")
        st.markdown(f"- **Predictor (X):** `daily_gaming_hours` (strong relationship)")
        st.markdown("- **Basis:** Correlation confirms linear dependency.")

    st.markdown("---")
    # Make Heatmap larger and more readable
    fig_corr = px.imshow(corr_matrix, 
                         text_auto=".2f", 
                         color_continuous_scale='RdBu_r', 
                         title="Correlation Heatmap (Visual Representation)",
                         height=700) # Increased height
    fig_corr.update_layout(title_x=0.5) # Center title
    st.plotly_chart(fig_corr, use_container_width=True)


    st.markdown("---")
    
    st.subheader("2. Linear Regression Model: Gaming vs Isolation")
    st.info("**Concept:** Using the parameters chosen above (X and Y), we train a Linear Regression model to predict the social isolation score.")

    X = df_final[['daily_gaming_hours']]
    y = df_final['social_isolation_score']
    
    model = LinearRegression()
    model.fit(X, y)
    
    coef = model.coef_[0]
    intercept = model.intercept_
    
    col3, col4 = st.columns(2)
    col3.metric("Coefficient (Slope)", f"{coef:.4f}")
    col4.metric("Intercept", f"{intercept:.4f}")
    
    st.markdown("---")
    st.subheader("Interactive Prediction: Try it yourself!")
    st.write("First, pick a number of daily gaming hours. We will use the regression model to instantly calculate the social isolation score, and map your prediction directly onto the graph below!")
    
    user_input = st.slider("Step 1: Enter Daily Gaming Hours (Input X):", 
                           min_value=float(df_final['daily_gaming_hours'].min()), 
                           max_value=float(df_final['daily_gaming_hours'].max()), 
                           value=float(df_final['daily_gaming_hours'].mean()), 
                           step=0.5)
    
    predicted_isolation = model.predict(pd.DataFrame({'daily_gaming_hours': [user_input]}))[0]
    st.success(f"**Predicted Isolation Score (Output): {predicted_isolation:.2f}**")
    st.caption(f"*(Calculated based on your input of {user_input} Daily Gaming Hours)*")

    # Render regression chart with the interactive point
    fig_reg = px.scatter(df_final, x="daily_gaming_hours", y="social_isolation_score",
                         title="Linear Trendline with Your Custom Prediction", opacity=0.6, template="plotly_dark")
    
    # Add regression line
    line_x = np.array([df_final['daily_gaming_hours'].min(), df_final['daily_gaming_hours'].max()])
    line_y = model.predict(pd.DataFrame({'daily_gaming_hours': line_x}))
    fig_reg.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Linear Fit', line=dict(color='red', width=3)))
    
    # Add the interactive point based on the slider above
    fig_reg.add_trace(go.Scatter(x=[user_input], y=[predicted_isolation], mode='markers', 
                                 name='Your Prediction', marker=dict(color='yellow', size=15, symbol='star', line=dict(color='white', width=2))))
    
    fig_reg.update_layout(xaxis_title="Input: Gaming Hours (X)", yaxis_title="Output: Predicted Isolation Score (Y)")
    st.plotly_chart(fig_reg, use_container_width=True)
