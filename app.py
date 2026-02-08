import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    </style>
""", unsafe_allow_html=True)

def load_data(uploaded_file):
    """Load and validate CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("⚠️ The uploaded file is empty. Please upload a valid CSV file.")
            return None
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading file: {str(e)}")
        return None

def get_column_types(df):
    """Categorize columns as numerical or categorical"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_cols, categorical_cols

def calculate_missing_values(df):
    """Calculate missing values for each column"""
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(
        'Missing %', ascending=False
    )
    return missing_data

def generate_insights(df, numerical_cols, categorical_cols, missing_data):
    """Generate top 3 insights from the data"""
    insights = []
    
    # Insight 1: Missing data analysis
    if not missing_data.empty:
        top_missing = missing_data.iloc[0]
        insights.append(
            f"📉 **Missing Data Alert**: Column '{top_missing['Column']}' has "
            f"{top_missing['Missing %']:.1f}% missing values ({int(top_missing['Missing Count'])} records)."
        )
    else:
        insights.append("✅ **Data Quality**: No missing values detected in the dataset!")
    
    # Insight 2: Categorical analysis (top category)
    if categorical_cols:
        for col in categorical_cols[:3]:
            value_counts = df[col].value_counts()
            if len(value_counts) > 0:
                top_category = value_counts.index[0]
                percentage = (value_counts.iloc[0] / len(df) * 100)
                if percentage > 20:
                    insights.append(
                        f"📊 **Top Category**: In '{col}', the value '{top_category}' "
                        f"appears {percentage:.1f}% of the time ({value_counts.iloc[0]} records)."
                    )
                    break
    
    # Insight 3: Numerical analysis
    if numerical_cols:
        for col in numerical_cols[:3]:
            if df[col].notna().sum() > 0:
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                
                insights.append(
                    f"📈 **Statistical Summary**: Column '{col}' has a mean of {mean_val:.2f}, "
                    f"median of {median_val:.2f}, and standard deviation of {std_val:.2f}."
                )
                break
    
    # Additional insight: Dataset size
    if len(insights) < 3:
        insights.append(
            f"📁 **Dataset Overview**: The dataset contains {len(df):,} rows and "
            f"{len(df.columns)} columns ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)."
        )
    
    return insights[:3]

def apply_filters(df, numerical_cols, categorical_cols):
    """Apply filters from sidebar and return filtered dataframe"""
    filtered_df = df.copy()
    
    st.sidebar.markdown("### 🔍 Filter Data")
    
    # Numerical filters
    if numerical_cols:
        st.sidebar.markdown("#### Numerical Filters")
        for col in numerical_cols:
            if filtered_df[col].notna().sum() > 0:
                min_val = float(filtered_df[col].min())
                max_val = float(filtered_df[col].max())
                
                if min_val != max_val:
                    selected_range = st.sidebar.slider(
                        f"{col}",
                        min_val,
                        max_val,
                        (min_val, max_val),
                        key=f"num_{col}"
                    )
                    filtered_df = filtered_df[
                        (filtered_df[col] >= selected_range[0]) & 
                        (filtered_df[col] <= selected_range[1])
                    ]
    
    # Categorical filters
    if categorical_cols:
        st.sidebar.markdown("#### Categorical Filters")
        for col in categorical_cols:
            unique_values = filtered_df[col].dropna().unique().tolist()
            if unique_values:
                selected_values = st.sidebar.multiselect(
                    f"{col}",
                    options=unique_values,
                    default=unique_values,
                    key=f"cat_{col}"
                )
                if selected_values:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    return filtered_df

def create_download_link(df):
    """Create download button for filtered data"""
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">📊 Business Analytics Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload a CSV file to analyze"
    )
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Get column types
            numerical_cols, categorical_cols = get_column_types(df)
            
            # Apply filters
            filtered_df = apply_filters(df, numerical_cols, categorical_cols)
            
            # Check if filters resulted in empty dataset
            if filtered_df.empty:
                st.warning("⚠️ No data matches the current filters. Please adjust your filter settings.")
                return
            
            # Main content area
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📄 Total Rows",
                    value=f"{len(df):,}",
                    delta=f"{len(filtered_df):,} filtered"
                )
            
            with col2:
                st.metric(
                    label="📊 Total Columns",
                    value=len(df.columns)
                )
            
            with col3:
                missing_count = df.isnull().sum().sum()
                st.metric(
                    label="❓ Missing Values",
                    value=f"{missing_count:,}"
                )
            
            # Missing Values Analysis
            st.markdown('<div class="section-header">📉 Missing Values Analysis</div>', 
                       unsafe_allow_html=True)
            
            missing_data = calculate_missing_values(df)
            
            if not missing_data.empty:
                fig_missing = px.bar(
                    missing_data,
                    x='Column',
                    y='Missing %',
                    title='Missing Values by Column (%)',
                    labels={'Missing %': 'Percentage Missing'},
                    color='Missing %',
                    color_continuous_scale='Reds'
                )
                fig_missing.update_layout(
                    xaxis_tickangle=-45,
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_missing, use_container_width=True)
                
                st.dataframe(
                    missing_data,
                    use_container_width=True
                )
            else:
                st.success("✅ No missing values detected in the dataset!")
            
            # Descriptive Statistics
            if numerical_cols:
                st.markdown('<div class="section-header">📈 Descriptive Statistics</div>', 
                           unsafe_allow_html=True)
                
                stats_df = filtered_df[numerical_cols].describe().T
                stats_df = stats_df[['mean', '50%', 'std', 'min', 'max']]
                stats_df.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
                stats_df = stats_df.round(2)
                
                st.dataframe(
                    stats_df,
                    use_container_width=True
                )
            
            # Data Preview
            st.markdown('<div class="section-header">📋 Filtered Data Preview</div>', 
                       unsafe_allow_html=True)
            
            st.info(f"Showing {len(filtered_df):,} of {len(df):,} rows ({len(filtered_df)/len(df)*100:.1f}%)")
            st.dataframe(filtered_df.head(100), use_container_width=True)
            
            # Visualizations
            st.markdown('<div class="section-header">📊 Data Visualizations</div>', 
                       unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            # Histogram for numerical columns
            with viz_col1:
                if numerical_cols:
                    st.markdown("#### 📊 Histogram (Numerical Data)")
                    selected_num_col = st.selectbox(
                        "Select numerical column",
                        numerical_cols,
                        key="hist_select"
                    )
                    
                    if selected_num_col:
                        fig_hist = px.histogram(
                            filtered_df,
                            x=selected_num_col,
                            nbins=30,
                            title=f'Distribution of {selected_num_col}',
                            labels={selected_num_col: selected_num_col},
                            color_discrete_sequence=['#1f77b4']
                        )
                        fig_hist.update_layout(
                            showlegend=False,
                            height=400,
                            bargap=0.1
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("No numerical columns available for histogram.")
            
            # Bar chart for categorical columns
            with viz_col2:
                if categorical_cols:
                    st.markdown("#### 📊 Bar Chart (Categorical Data)")
                    selected_cat_col = st.selectbox(
                        "Select categorical column",
                        categorical_cols,
                        key="bar_select"
                    )
                    
                    if selected_cat_col:
                        cat_counts = filtered_df[selected_cat_col].value_counts().head(15)
                        
                        fig_bar = px.bar(
                            x=cat_counts.index,
                            y=cat_counts.values,
                            title=f'Top Categories in {selected_cat_col}',
                            labels={'x': selected_cat_col, 'y': 'Count'},
                            color=cat_counts.values,
                            color_continuous_scale='Viridis'
                        )
                        fig_bar.update_layout(
                            showlegend=False,
                            height=400,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No categorical columns available for bar chart.")
            
            # Key Findings Section
            st.markdown('<div class="section-header">💡 Key Findings</div>', 
                       unsafe_allow_html=True)
            
            insights = generate_insights(df, numerical_cols, categorical_cols, missing_data)
            
            for i, insight in enumerate(insights, 1):
                st.markdown(
                    f'<div class="insight-box"><strong>Insight {i}:</strong> {insight}</div>',
                    unsafe_allow_html=True
                )
            
            # Download Section
            st.markdown('<div class="section-header">💾 Export Data</div>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                csv_data = create_download_link(filtered_df)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name=f"filtered_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        # Welcome screen
        st.markdown("""
        ### Welcome to the Business Analytics Dashboard! 👋
        
        This interactive dashboard helps you analyze your business data with ease.
        
        #### 🚀 Features:
        - **📤 Upload Data**: Support for any CSV file format
        - **📊 Summary Statistics**: Automatic calculation of key metrics
        - **🔍 Smart Filtering**: Filter data by numerical and categorical columns
        - **📈 Interactive Visualizations**: Dynamic histograms and bar charts
        - **💡 AI-Driven Insights**: Automatically generated findings
        - **💾 Export Results**: Download filtered data instantly
        
        #### 📝 How to Get Started:
        1. Click on **"Browse files"** in the sidebar
        2. Upload your CSV file
        3. Explore the automatic analysis and insights
        4. Use filters to drill down into your data
        5. Generate visualizations and export results
        
        ---
        **Ready to begin?** Upload your CSV file using the sidebar! 👈
        """)
        
        with st.expander("📋 Example CSV Structure"):
            sample_df = pd.DataFrame({
                'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'Product': ['Widget A', 'Widget B', 'Widget A'],
                'Sales': [1500, 2300, 1800],
                'Quantity': [10, 15, 12],
                'Region': ['North', 'South', 'North']
            })
            st.dataframe(sample_df, use_container_width=True)
            st.caption("Your CSV can have any number of columns with different data types!")

if __name__ == "__main__":
    main()
