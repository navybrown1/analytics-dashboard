import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import numpy as np
from datetime import datetime, timedelta
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Business Analytics Dashboard - Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
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
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================
@st.cache_data
def load_data(uploaded_file):
    """Load and validate uploaded CSV file with caching"""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            return None, "The uploaded file is empty."
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def compute_statistics(df, numerical_cols):
    """Cache heavy statistical computations"""
    stats = {}
    for col in numerical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats[col] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'q1': col_data.quantile(0.25),
                'q3': col_data.quantile(0.75),
                'skew': col_data.skew(),
                'kurtosis': col_data.kurtosis(),
                'count': len(col_data),
                'missing': df[col].isnull().sum()
            }
    return stats

@st.cache_data
def compute_correlation_matrix(df, numerical_cols):
    """Cache correlation matrix computation"""
    if len(numerical_cols) >= 2:
        return df[numerical_cols].corr()
    return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_column_types(df):
    """Categorize columns by data type"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to detect date columns stored as strings
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col].dropna().head(10))
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except (ValueError, TypeError):
            pass
    
    return numerical_cols, categorical_cols, datetime_cols

def calculate_missing_values(df):
    """Calculate missing values analysis"""
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data Type': df.dtypes.values.astype(str)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(
        'Missing %', ascending=False
    ).reset_index(drop=True)
    return missing_data

def detect_outliers(df, numerical_cols):
    """Detect outliers using IQR method"""
    outlier_info = {}
    for col in numerical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(col_data) * 100),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_outlier': outliers.min() if len(outliers) > 0 else None,
                'max_outlier': outliers.max() if len(outliers) > 0 else None
            }
    return outlier_info

def generate_advanced_insights(df, numerical_cols, categorical_cols, missing_data, outlier_info, stats):
    """Generate comprehensive AI-driven insights"""
    insights = []
    
    # Insight 1: Data quality score
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    quality_score = ((total_cells - missing_cells) / total_cells * 100)
    quality_label = "Excellent" if quality_score >= 95 else "Good" if quality_score >= 85 else "Needs Attention"
    insights.append({
        'icon': '🎯',
        'title': 'Data Quality Score',
        'description': f"Overall data quality is **{quality_label}** at {quality_score:.1f}%. "
                       f"({missing_cells:,} missing values across {total_cells:,} total data points)",
        'type': 'quality'
    })
    
    # Insight 2: Missing data patterns
    if not missing_data.empty:
        top_missing = missing_data.iloc[0]
        insights.append({
            'icon': '📉',
            'title': 'Missing Data Alert',
            'description': f"Column **'{top_missing['Column']}'** has the highest missing rate "
                          f"at {top_missing['Missing %']:.1f}% ({int(top_missing['Missing Count'])} records). "
                          f"Consider imputation or removal.",
            'type': 'warning'
        })
    
    # Insight 3: Outlier detection
    if outlier_info:
        max_outlier_col = max(outlier_info.items(), key=lambda x: x[1]['count'])
        if max_outlier_col[1]['count'] > 0:
            insights.append({
                'icon': '⚠️',
                'title': 'Outlier Detection',
                'description': f"Column **'{max_outlier_col[0]}'** has {max_outlier_col[1]['count']} "
                              f"outliers ({max_outlier_col[1]['percentage']:.1f}% of data). "
                              f"Range: [{max_outlier_col[1]['lower_bound']:.2f}, {max_outlier_col[1]['upper_bound']:.2f}]",
                'type': 'warning'
            })
    
    # Insight 4: Distribution shape
    if stats:
        for col, col_stats in stats.items():
            skew = col_stats.get('skew', 0)
            if abs(skew) > 1:
                direction = "right-skewed (positive)" if skew > 0 else "left-skewed (negative)"
                insights.append({
                    'icon': '📊',
                    'title': f'Distribution Alert: {col}',
                    'description': f"Column **'{col}'** is significantly {direction} "
                                  f"(skewness: {skew:.2f}). Consider log transformation for analysis.",
                    'type': 'info'
                })
                break
    
    # Insight 5: Categorical dominance
    if categorical_cols:
        for col in categorical_cols[:3]:
            value_counts = df[col].value_counts()
            if len(value_counts) > 0:
                top_pct = (value_counts.iloc[0] / len(df) * 100)
                if top_pct > 30:
                    insights.append({
                        'icon': '📋',
                        'title': f'Category Dominance: {col}',
                        'description': f"In **'{col}'**, the value '{value_counts.index[0]}' "
                                      f"dominates with {top_pct:.1f}% of all records. "
                                      f"({len(value_counts)} unique categories total)",
                        'type': 'info'
                    })
                    break
    
    # Insight 6: Dataset summary
    insights.append({
        'icon': '📁',
        'title': 'Dataset Overview',
        'description': f"Dataset contains **{len(df):,} rows** and **{len(df.columns)} columns** "
                       f"({len(numerical_cols)} numerical, {len(categorical_cols)} categorical). "
                       f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB",
        'type': 'success'
    })
    
    return insights[:6]

def apply_advanced_filters(df, numerical_cols, categorical_cols, datetime_cols):
    """Advanced filtering with sidebar controls"""
    filtered_df = df.copy()
    
    st.sidebar.markdown("### 🔍 Filter Data")
    
    # Row limit filter
    row_limit = st.sidebar.number_input(
        "Max rows to display",
        min_value=10,
        max_value=len(df),
        value=min(1000, len(df)),
        step=100,
        key="row_limit"
    )
    
    # Numerical filters
    if numerical_cols:
        with st.sidebar.expander("📊 Numerical Filters", expanded=False):
            for col in numerical_cols:
                if filtered_df[col].notna().sum() > 0:
                    min_val = float(filtered_df[col].min())
                    max_val = float(filtered_df[col].max())
                    
                    if min_val != max_val:
                        selected_range = st.slider(
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
        with st.sidebar.expander("📋 Categorical Filters", expanded=False):
            for col in categorical_cols:
                unique_values = filtered_df[col].dropna().unique().tolist()
                if unique_values and len(unique_values) <= 50:
                    selected_values = st.multiselect(
                        f"{col}",
                        options=unique_values,
                        default=unique_values,
                        key=f"cat_{col}"
                    )
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    # Date filters
    if datetime_cols:
        with st.sidebar.expander("📅 Date Filters", expanded=False):
            for col in datetime_cols:
                try:
                    date_series = pd.to_datetime(filtered_df[col], errors='coerce')
                    valid_dates = date_series.dropna()
                    if len(valid_dates) > 0:
                        min_date = valid_dates.min().date()
                        max_date = valid_dates.max().date()
                        if min_date != max_date:
                            date_range = st.date_input(
                                f"{col} range",
                                value=(min_date, max_date),
                                min_value=min_date,
                                max_value=max_date,
                                key=f"date_{col}"
                            )
                            if len(date_range) == 2:
                                mask = (date_series >= pd.Timestamp(date_range[0])) & \
                                       (date_series <= pd.Timestamp(date_range[1]))
                                filtered_df = filtered_df[mask]
                except Exception:
                    pass
    
    return filtered_df, row_limit

def create_export_data(df, format_type='csv'):
    """Create exportable data in multiple formats"""
    output = BytesIO()
    if format_type == 'csv':
        df.to_csv(output, index=False)
        mime_type = 'text/csv'
        extension = 'csv'
    elif format_type == 'excel':
        df.to_excel(output, index=False, engine='openpyxl')
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        extension = 'xlsx'
    elif format_type == 'json':
        output.write(df.to_json(orient='records', indent=2).encode())
        mime_type = 'application/json'
        extension = 'json'
    else:
        df.to_csv(output, index=False)
        mime_type = 'text/csv'
        extension = 'csv'
    output.seek(0)
    return output.getvalue(), mime_type, extension

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_histogram(df, col, nbins=30):
    """Create an interactive histogram"""
    fig = px.histogram(
        df, x=col, nbins=nbins,
        title=f'Distribution of {col}',
        color_discrete_sequence=['#1f77b4'],
        marginal='box'
    )
    fig.update_layout(showlegend=False, height=450, bargap=0.1)
    return fig

def create_bar_chart(df, col, top_n=15):
    """Create a bar chart for categorical data"""
    counts = df[col].value_counts().head(top_n)
    fig = px.bar(
        x=counts.index, y=counts.values,
        title=f'Top {min(top_n, len(counts))} Categories in {col}',
        labels={'x': col, 'y': 'Count'},
        color=counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False, height=450, xaxis_tickangle=-45)
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None):
    """Create an interactive scatter plot"""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=f'{y_col} vs {x_col}',
        opacity=0.7,
        trendline='ols' if color_col is None else None
    )
    fig.update_layout(height=450)
    return fig

def create_box_plot(df, numerical_col, categorical_col=None):
    """Create a box plot"""
    if categorical_col:
        fig = px.box(
            df, x=categorical_col, y=numerical_col,
            title=f'{numerical_col} by {categorical_col}',
            color=categorical_col
        )
    else:
        fig = px.box(
            df, y=numerical_col,
            title=f'Box Plot of {numerical_col}'
        )
    fig.update_layout(height=450, showlegend=False)
    return fig

def create_pie_chart(df, col, top_n=10):
    """Create a pie chart"""
    counts = df[col].value_counts().head(top_n)
    fig = px.pie(
        values=counts.values, names=counts.index,
        title=f'Distribution of {col}',
        hole=0.3
    )
    fig.update_layout(height=450)
    return fig

def create_heatmap(corr_matrix):
    """Create a correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig.update_layout(
        title='Correlation Matrix',
        height=500,
        xaxis_tickangle=-45
    )
    return fig

def create_line_chart(df, x_col, y_col, color_col=None):
    """Create a line chart (useful for time series)"""
    fig = px.line(
        df.sort_values(x_col), x=x_col, y=y_col, color=color_col,
        title=f'{y_col} over {x_col}',
        markers=True
    )
    fig.update_layout(height=450)
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown(
        '<div class="main-header">📊 Business Analytics Dashboard - Enhanced</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Theme selection
    color_theme = st.sidebar.selectbox(
        "🎨 Color Theme",
        ["Default Blue", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
        key="color_theme"
    )
    theme_map = {
        "Default Blue": "Blues", "Viridis": "Viridis", "Plasma": "Plasma",
        "Inferno": "Inferno", "Magma": "Magma", "Cividis": "Cividis"
    }
    selected_cmap = theme_map.get(color_theme, "Blues")
    
    st.sidebar.markdown("---")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📤 Upload CSV File",
        type=['csv'],
        help="Upload a CSV file to analyze. Max size: 200MB."
    )
    
    if uploaded_file is not None:
        # Load data with caching
        df, error = load_data(uploaded_file)
        
        if error:
            st.error(f"⚠️ Error loading file: {error}")
            return
        
        if df is None:
            st.error("⚠️ The uploaded file is empty.")
            return
        
        # Store original data info
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # Get column types
        numerical_cols, categorical_cols, datetime_cols = get_column_types(df)
        
        # Apply filters
        filtered_df, row_limit = apply_advanced_filters(
            df, numerical_cols, categorical_cols, datetime_cols
        )
        
        if filtered_df.empty:
            st.warning("⚠️ No data matches the current filters. Please adjust your filter settings.")
            return
        
        # Compute statistics (cached)
        stats = compute_statistics(df, numerical_cols)
        corr_matrix = compute_correlation_matrix(filtered_df, numerical_cols)
        missing_data = calculate_missing_values(df)
        outlier_info = detect_outliers(df, numerical_cols)
        
        # ================================================================
        # KEY METRICS ROW
        # ================================================================
        st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
        
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            st.metric(
                label="📄 Total Rows",
                value=f"{original_rows:,}",
                delta=f"{len(filtered_df):,} after filters"
            )
        
        with metric_cols[1]:
            st.metric(
                label="📊 Columns",
                value=original_cols,
                delta=f"{len(numerical_cols)} numeric"
            )
        
        with metric_cols[2]:
            missing_count = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            st.metric(
                label="❓ Missing Values",
                value=f"{missing_count:,}",
                delta=f"{(missing_count/total_cells*100):.1f}%"
            )
        
        with metric_cols[3]:
            total_outliers = sum(info['count'] for info in outlier_info.values())
            st.metric(
                label="⚠️ Outliers",
                value=f"{total_outliers:,}"
            )
        
        with metric_cols[4]:
            quality_score = ((total_cells - missing_count) / total_cells * 100)
            st.metric(
                label="✅ Quality Score",
                value=f"{quality_score:.1f}%"
            )
        
        # ================================================================
        # TABBED INTERFACE
        # ================================================================
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Data Overview",
            "📈 Statistics",
            "📊 Visualizations",
            "🔗 Correlations",
            "💡 Insights",
            "💾 Export"
        ])
        
        # ============================================================
        # TAB 1: DATA OVERVIEW
        # ============================================================
        with tab1:
            st.markdown('<div class="section-header">📋 Data Overview</div>',
                       unsafe_allow_html=True)
            
            # Data types summary
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null': df.isnull().sum().values,
                    'Unique': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True, height=350)
            
            with col_b:
                st.markdown("#### Missing Values")
                if not missing_data.empty:
                    fig_missing = px.bar(
                        missing_data,
                        x='Column', y='Missing %',
                        title='Missing Values by Column (%)',
                        color='Missing %',
                        color_continuous_scale='Reds'
                    )
                    fig_missing.update_layout(
                        xaxis_tickangle=-45, height=350, showlegend=False
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.success("✅ No missing values detected!")
            
            # Data preview
            st.markdown("#### Filtered Data Preview")
            st.info(
                f"Showing {min(row_limit, len(filtered_df)):,} of "
                f"{len(filtered_df):,} filtered rows "
                f"({len(filtered_df)/original_rows*100:.1f}% of original {original_rows:,})"
            )
            st.dataframe(filtered_df.head(row_limit), use_container_width=True)
        
        # ============================================================
        # TAB 2: STATISTICS
        # ============================================================
        with tab2:
            st.markdown('<div class="section-header">📈 Descriptive Statistics</div>',
                       unsafe_allow_html=True)
            
            if numerical_cols:
                # Summary statistics table
                stats_df = filtered_df[numerical_cols].describe().T
                stats_df = stats_df.round(2)
                
                # Add additional statistics
                for col in numerical_cols:
                    col_data = filtered_df[col].dropna()
                    if len(col_data) > 0:
                        stats_df.loc[col, 'skew'] = col_data.skew()
                        stats_df.loc[col, 'kurtosis'] = col_data.kurtosis()
                        stats_df.loc[col, 'missing'] = filtered_df[col].isnull().sum()
                
                stats_df = stats_df.round(2)
                st.dataframe(
                    stats_df,
                    use_container_width=True
                )
                
                # Outlier detection section
                st.markdown("#### ⚠️ Outlier Detection (IQR Method)")
                
                outlier_data = []
                for col, info in outlier_info.items():
                    outlier_data.append({
                        'Column': col,
                        'Outlier Count': info['count'],
                        'Outlier %': round(info['percentage'], 2),
                        'Lower Bound': round(info['lower_bound'], 2),
                        'Upper Bound': round(info['upper_bound'], 2)
                    })
                
                if outlier_data:
                    outlier_df = pd.DataFrame(outlier_data)
                    outlier_df = outlier_df.sort_values('Outlier Count', ascending=False)
                    st.dataframe(outlier_df, use_container_width=True)
                    
                    # Outlier visualization
                    outlier_col = st.selectbox(
                        "Select column for outlier visualization",
                        numerical_cols,
                        key="outlier_viz_col"
                    )
                    if outlier_col:
                        fig_box = create_box_plot(filtered_df, outlier_col)
                        st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.success("✅ No outliers detected in numerical columns!")
            else:
                st.info("No numerical columns available for statistical analysis.")
        
        # ============================================================
        # TAB 3: VISUALIZATIONS
        # ============================================================
        with tab3:
            st.markdown('<div class="section-header">📊 Interactive Visualizations</div>',
                       unsafe_allow_html=True)
            
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot",
                 "Pie Chart", "Line Chart", "Grouped Analysis"],
                key="viz_type"
            )
            
            # --- Histogram ---
            if viz_type == "Histogram":
                if numerical_cols:
                    col_a, col_b = st.columns([3, 1])
                    with col_b:
                        hist_col = st.selectbox("Column", numerical_cols, key="hist_col")
                        nbins = st.slider("Bins", 5, 100, 30, key="hist_bins")
                    with col_a:
                        if hist_col:
                            fig = create_histogram(filtered_df, hist_col, nbins)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numerical columns for histogram.")
            
            # --- Bar Chart ---
            elif viz_type == "Bar Chart":
                if categorical_cols:
                    col_a, col_b = st.columns([3, 1])
                    with col_b:
                        bar_col = st.selectbox("Column", categorical_cols, key="bar_col")
                        top_n = st.slider("Top N", 5, 30, 15, key="bar_top_n")
                    with col_a:
                        if bar_col:
                            fig = create_bar_chart(filtered_df, bar_col, top_n)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No categorical columns for bar chart.")
            
            # --- Scatter Plot ---
            elif viz_type == "Scatter Plot":
                if len(numerical_cols) >= 2:
                    col_a, col_b = st.columns([3, 1])
                    with col_b:
                        x_col = st.selectbox("X Axis", numerical_cols, key="scatter_x")
                        y_options = [c for c in numerical_cols if c != x_col]
                        y_col = st.selectbox(
                            "Y Axis",
                            y_options if y_options else numerical_cols,
                            key="scatter_y"
                        )
                        color_options = ["None"] + categorical_cols
                        color_col = st.selectbox("Color By", color_options, key="scatter_color")
                        color_col = None if color_col == "None" else color_col
                    with col_a:
                        if x_col and y_col:
                            fig = create_scatter_plot(filtered_df, x_col, y_col, color_col)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numerical columns for scatter plot.")
            
            # --- Box Plot ---
            elif viz_type == "Box Plot":
                if numerical_cols:
                    col_a, col_b = st.columns([3, 1])
                    with col_b:
                        box_num = st.selectbox("Numerical Column", numerical_cols, key="box_num")
                        group_options = ["None"] + categorical_cols
                        box_cat = st.selectbox("Group By", group_options, key="box_cat")
                        box_cat = None if box_cat == "None" else box_cat
                    with col_a:
                        if box_num:
                            fig = create_box_plot(filtered_df, box_num, box_cat)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numerical columns for box plot.")
            
            # --- Pie Chart ---
            elif viz_type == "Pie Chart":
                if categorical_cols:
                    col_a, col_b = st.columns([3, 1])
                    with col_b:
                        pie_col = st.selectbox("Column", categorical_cols, key="pie_col")
                        pie_top = st.slider("Top N", 3, 20, 10, key="pie_top")
                    with col_a:
                        if pie_col:
                            fig = create_pie_chart(filtered_df, pie_col, pie_top)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No categorical columns for pie chart.")
            
            # --- Line Chart ---
            elif viz_type == "Line Chart":
                all_cols = datetime_cols + categorical_cols + numerical_cols
                if all_cols and numerical_cols:
                    col_a, col_b = st.columns([3, 1])
                    with col_b:
                        line_x = st.selectbox("X Axis", all_cols, key="line_x")
                        line_y = st.selectbox("Y Axis", numerical_cols, key="line_y")
                        line_color_opts = ["None"] + categorical_cols
                        line_color = st.selectbox("Color By", line_color_opts, key="line_color")
                        line_color = None if line_color == "None" else line_color
                    with col_a:
                        if line_x and line_y:
                            fig = create_line_chart(filtered_df, line_x, line_y, line_color)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need columns for line chart.")
            
            # --- Grouped Analysis ---
            elif viz_type == "Grouped Analysis":
                if categorical_cols and numerical_cols:
                    col_a, col_b = st.columns([3, 1])
                    with col_b:
                        group_col = st.selectbox("Group By", categorical_cols, key="group_col")
                        agg_col = st.selectbox("Aggregate Column", numerical_cols, key="agg_col")
                        agg_func = st.selectbox(
                            "Aggregation",
                            ["mean", "sum", "count", "median", "min", "max", "std"],
                            key="agg_func"
                        )
                    with col_a:
                        if group_col and agg_col:
                            grouped = filtered_df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                            grouped.columns = [group_col, f'{agg_func}({agg_col})']
                            grouped = grouped.sort_values(f'{agg_func}({agg_col})', ascending=False)
                            
                            fig = px.bar(
                                grouped,
                                x=group_col,
                                y=f'{agg_func}({agg_col})',
                                title=f'{agg_func.title()} of {agg_col} by {group_col}',
                                color=f'{agg_func}({agg_col})',
                                color_continuous_scale=selected_cmap
                            )
                            fig.update_layout(height=450, xaxis_tickangle=-45, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(grouped, use_container_width=True)
                else:
                    st.info("Need both categorical and numerical columns for grouped analysis.")
        
        # ============================================================
        # TAB 4: CORRELATIONS
        # ============================================================
        with tab4:
            st.markdown('<div class="section-header">🔗 Correlation Analysis</div>',
                       unsafe_allow_html=True)
            
            if corr_matrix is not None and len(numerical_cols) >= 2:
                # Correlation heatmap
                fig_corr = create_heatmap(corr_matrix)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Top correlations
                st.markdown("#### Top Correlation Pairs")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Column 1': corr_matrix.columns[i],
                            'Column 2': corr_matrix.columns[j],
                            'Correlation': round(corr_matrix.iloc[i, j], 4),
                            'Strength': 'Strong' if abs(corr_matrix.iloc[i, j]) > 0.7
                                        else 'Moderate' if abs(corr_matrix.iloc[i, j]) > 0.4
                                        else 'Weak'
                        })
                
                corr_pairs_df = pd.DataFrame(corr_pairs).sort_values(
                    'Correlation', ascending=False, key=abs
                )
                st.dataframe(corr_pairs_df, use_container_width=True)
                
                # Interactive scatter for top correlation
                st.markdown("#### Explore Correlation")
                corr_col1 = st.selectbox("Column 1", numerical_cols, key="corr_col1")
                remaining = [c for c in numerical_cols if c != corr_col1]
                corr_col2 = st.selectbox(
                    "Column 2",
                    remaining if remaining else numerical_cols,
                    key="corr_col2"
                )
                
                if corr_col1 and corr_col2:
                    fig_scatter = create_scatter_plot(filtered_df, corr_col1, corr_col2)
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Need at least 2 numerical columns for correlation analysis.")
        
        # ============================================================
        # TAB 5: INSIGHTS
        # ============================================================
        with tab5:
            st.markdown('<div class="section-header">💡 AI-Driven Insights</div>',
                       unsafe_allow_html=True)
            
            insights = generate_advanced_insights(
                df, numerical_cols, categorical_cols,
                missing_data, outlier_info, stats
            )
            
            for i, insight in enumerate(insights, 1):
                box_class = {
                    'quality': 'insight-box',
                    'warning': 'warning-box',
                    'info': 'insight-box',
                    'success': 'success-box'
                }.get(insight['type'], 'insight-box')
                
                st.markdown(
                    f'<div class="{box_class}">'
                    f'<strong>{insight["icon"]} Insight {i}: {insight["title"]}</strong><br>'
                    f'{insight["description"]}</div>',
                    unsafe_allow_html=True
                )
            
            # Data profiling summary
            st.markdown("#### 📊 Quick Data Profile")
            
            profile_cols = st.columns(3)
            
            with profile_cols[0]:
                st.markdown("**Numerical Columns**")
                for col in numerical_cols:
                    col_data = filtered_df[col].dropna()
                    if len(col_data) > 0:
                        st.markdown(
                            f"- **{col}**: {col_data.min():.2f} to {col_data.max():.2f} "
                            f"(avg: {col_data.mean():.2f})"
                        )
            
            with profile_cols[1]:
                st.markdown("**Categorical Columns**")
                for col in categorical_cols:
                    n_unique = filtered_df[col].nunique()
                    top_val = filtered_df[col].mode()
                    top_val_str = top_val.iloc[0] if len(top_val) > 0 else "N/A"
                    st.markdown(f"- **{col}**: {n_unique} unique (top: {top_val_str})")
            
            with profile_cols[2]:
                st.markdown("**Data Quality**")
                for col in df.columns:
                    missing_pct = (df[col].isnull().sum() / len(df) * 100)
                    status = "✅" if missing_pct == 0 else "⚠️" if missing_pct < 10 else "❌"
                    st.markdown(f"- {status} **{col}**: {missing_pct:.1f}% missing")
        
        # ============================================================
        # TAB 6: EXPORT
        # ============================================================
        with tab6:
            st.markdown('<div class="section-header">💾 Export Data</div>',
                       unsafe_allow_html=True)
            
            st.markdown(
                f"**Filtered dataset**: {len(filtered_df):,} rows x {len(filtered_df.columns)} columns"
            )
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            with export_col1:
                csv_data, csv_mime, csv_ext = create_export_data(filtered_df, 'csv')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"analytics_export_{timestamp}.{csv_ext}",
                    mime=csv_mime,
                    use_container_width=True
                )
            
            with export_col2:
                try:
                    excel_data, excel_mime, excel_ext = create_export_data(filtered_df, 'excel')
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"analytics_export_{timestamp}.{excel_ext}",
                        mime=excel_mime,
                        use_container_width=True
                    )
                except Exception:
                    st.warning("Excel export requires openpyxl. Install with: pip install openpyxl")
            
            with export_col3:
                json_data, json_mime, json_ext = create_export_data(filtered_df, 'json')
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"analytics_export_{timestamp}.{json_ext}",
                    mime=json_mime,
                    use_container_width=True
                )
            
            # Statistics export
            st.markdown("---")
            st.markdown("#### Export Statistics Report")
            
            if numerical_cols:
                stats_report = filtered_df[numerical_cols].describe().T.round(2)
                stats_csv, _, _ = create_export_data(stats_report.reset_index(), 'csv')
                st.download_button(
                    label="📊 Download Statistics Report (CSV)",
                    data=stats_csv,
                    file_name=f"statistics_report_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        # ================================================================
        # WELCOME SCREEN
        # ================================================================
        st.markdown("""
        ### Welcome to the Enhanced Business Analytics Dashboard! 👋
        
        This **production-ready** dashboard helps you analyze any CSV dataset
        with powerful visualizations and AI-driven insights.
        
        ---
        
        #### 🚀 Key Features:
        
        | Feature | Description |
        |---------|-------------|
        | 📤 **Data Upload** | CSV file support with validation |
        | 📊 **7 Chart Types** | Histogram, Bar, Scatter, Box, Pie, Line, Grouped |
        | 🔍 **Smart Filtering** | Numerical, categorical, and date filters |
        | 📈 **Statistics** | Descriptive stats, skew, kurtosis, outliers |
        | 🔗 **Correlations** | Heatmap and pairwise analysis |
        | 💡 **AI Insights** | Automated data quality and pattern detection |
        | ⚠️ **Outlier Detection** | IQR-based anomaly identification |
        | 💾 **Multi-Format Export** | CSV, Excel, and JSON downloads |
        | 🎨 **Themes** | 6 color themes to choose from |
        | ⚡ **Performance** | Cached computations for speed |
        
        ---
        
        #### 📝 Getting Started:
        1. Click **"Browse files"** in the sidebar
        2. Upload your CSV file
        3. Explore tabs: Overview, Statistics, Visualizations, Correlations, Insights
        4. Apply filters to focus on specific data segments
        5. Export results in your preferred format
        
        **Ready to begin?** Upload your CSV file using the sidebar! 👈
        """)
        
        # Sample data preview
        with st.expander("📋 Sample CSV Structure"):
            sample_df = pd.DataFrame({
                'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'Product': ['Widget A', 'Widget B', 'Widget A'],
                'Category': ['Electronics', 'Electronics', 'Home'],
                'Sales': [1500, 2300, 1800],
                'Quantity': [10, 15, 12],
                'Region': ['North', 'South', 'North'],
                'Revenue': [15000, 34500, 21600],
                'Profit': [6000, 13800, 8640]
            })
            st.dataframe(sample_df, use_container_width=True)
            st.caption(
                "Your CSV can have any columns! The dashboard auto-detects "
                "numerical, categorical, and date columns."
            )
        
        with st.expander("💡 Tips for Best Results"):
            st.markdown("""
            - **Clean headers**: Use descriptive column names without special characters
            - **Consistent formats**: Keep dates in YYYY-MM-DD format
            - **Data types**: Numerical columns should contain only numbers
            - **Missing values**: Leave cells empty (not "N/A" or "null" strings)
            - **File size**: Works best with files under 50MB for fast analysis
            """)

if __name__ == "__main__":
    main()
