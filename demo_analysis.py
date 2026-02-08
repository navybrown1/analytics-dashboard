"""
Demo Analysis Script for Business Analytics Dashboard
======================================================
This script demonstrates the analytical capabilities
without requiring a Streamlit server.

Run with: python demo_analysis.py
"""

import pandas as pd
import numpy as np
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILE = os.path.join(SCRIPT_DIR, "sample_data.csv")
LARGE_FILE = os.path.join(SCRIPT_DIR, "sample_data_large.csv")


def print_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subheader(title):
    """Print a formatted sub-header"""
    print(f"\n--- {title} ---")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_and_inspect(filepath):
    """Load CSV and display basic information"""
    print_header("📁 DATA LOADING & INSPECTION")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    print(f"\n✅ File loaded: {os.path.basename(filepath)}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    print_subheader("Column Information")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        unique = df[col].nunique()
        print(f"   {col:20s} | {str(dtype):10s} | {non_null:4d} non-null | "
              f"{null_count:2d} missing | {unique:3d} unique")
    
    print_subheader("First 5 Rows")
    print(df.head().to_string(index=False))
    
    return df


def analyze_missing_data(df):
    """Analyze missing values"""
    print_header("❓ MISSING VALUES ANALYSIS")
    
    missing = df.isnull().sum()
    total = len(df)
    
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            has_missing = True
            pct = (missing[col] / total * 100)
            print(f"   ⚠️  {col:20s}: {missing[col]:3d} missing ({pct:.1f}%)")
    
    if not has_missing:
        print("   ✅ No missing values detected!")
    
    total_missing = missing.sum()
    total_cells = df.shape[0] * df.shape[1]
    quality_score = ((total_cells - total_missing) / total_cells * 100)
    print(f"\n   📊 Data Quality Score: {quality_score:.1f}%")
    print(f"   📊 Total missing cells: {total_missing:,} / {total_cells:,}")


def analyze_statistics(df):
    """Display descriptive statistics"""
    print_header("📈 DESCRIPTIVE STATISTICS")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        print("   No numerical columns found.")
        return
    
    for col in numerical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            print_subheader(f"Column: {col}")
            print(f"   Count:    {len(col_data):,.0f}")
            print(f"   Mean:     {col_data.mean():,.2f}")
            print(f"   Median:   {col_data.median():,.2f}")
            print(f"   Std Dev:  {col_data.std():,.2f}")
            print(f"   Min:      {col_data.min():,.2f}")
            print(f"   Max:      {col_data.max():,.2f}")
            print(f"   Q1 (25%): {col_data.quantile(0.25):,.2f}")
            print(f"   Q3 (75%): {col_data.quantile(0.75):,.2f}")
            print(f"   Skewness: {col_data.skew():.4f}")
            print(f"   Kurtosis: {col_data.kurtosis():.4f}")


def analyze_categories(df):
    """Analyze categorical columns"""
    print_header("📋 CATEGORICAL ANALYSIS")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        print("   No categorical columns found.")
        return
    
    for col in categorical_cols:
        print_subheader(f"Column: {col}")
        counts = df[col].value_counts()
        total = len(df)
        
        for value, count in counts.head(10).items():
            pct = (count / total * 100)
            bar = "█" * int(pct / 2)
            print(f"   {str(value):20s}: {count:4d} ({pct:5.1f}%) {bar}")


def analyze_outliers(df):
    """Detect outliers using IQR method"""
    print_header("⚠️ OUTLIER DETECTION (IQR Method)")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower) | (col_data > upper)]
            
            if len(outliers) > 0:
                print(f"\n   ⚠️  {col}: {len(outliers)} outliers detected")
                print(f"      Bounds: [{lower:.2f}, {upper:.2f}]")
                print(f"      Outlier range: [{outliers.min():.2f}, {outliers.max():.2f}]")
            else:
                print(f"   ✅ {col}: No outliers detected")


def analyze_correlations(df):
    """Compute correlation matrix"""
    print_header("🔗 CORRELATION ANALYSIS")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        print("   Need at least 2 numerical columns for correlation analysis.")
        return
    
    corr = df[numerical_cols].corr()
    
    print_subheader("Correlation Matrix")
    print(corr.round(3).to_string())
    
    # Find top correlations
    print_subheader("Notable Correlations")
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) > 0.3:
                strength = "Strong" if abs(val) > 0.7 else "Moderate"
                direction = "positive" if val > 0 else "negative"
                print(f"   {corr.columns[i]} ↔ {corr.columns[j]}: "
                      f"{val:.3f} ({strength} {direction})")


def generate_insights(df):
    """Generate automated insights"""
    print_header("💡 AUTO-GENERATED INSIGHTS")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    insights = []
    
    # Data quality
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    quality = ((total_cells - missing_cells) / total_cells * 100)
    label = "Excellent" if quality >= 95 else "Good" if quality >= 85 else "Needs Attention"
    insights.append(f"🎯 Data Quality: {label} ({quality:.1f}%)")
    
    # Top numerical insight
    if numerical_cols:
        col = numerical_cols[0]
        mean_val = df[col].mean()
        std_val = df[col].std()
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0
        variability = "high" if cv > 50 else "moderate" if cv > 20 else "low"
        insights.append(
            f"📈 '{col}' shows {variability} variability (CV: {cv:.1f}%, "
            f"mean: {mean_val:.2f}, std: {std_val:.2f})"
        )
    
    # Top categorical insight
    if categorical_cols:
        col = categorical_cols[0]
        top = df[col].value_counts().head(1)
        if len(top) > 0:
            pct = (top.iloc[0] / len(df) * 100)
            insights.append(
                f"📊 In '{col}', '{top.index[0]}' is most common ({pct:.1f}%)"
            )
    
    # Dataset size
    insights.append(
        f"📁 Dataset: {len(df):,} rows, {len(df.columns)} columns "
        f"({len(numerical_cols)} numeric, {len(categorical_cols)} categorical)"
    )
    
    for i, insight in enumerate(insights, 1):
        print(f"\n   Insight {i}: {insight}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "🔷" * 30)
    print("  📊 BUSINESS ANALYTICS - DEMO ANALYSIS")
    print("🔷" * 30)
    print(f"\nPython: {sys.version.split()[0]}")
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy:  {np.__version__}")
    
    # Choose file
    if os.path.exists(LARGE_FILE):
        filepath = LARGE_FILE
        print(f"\nUsing large dataset: {os.path.basename(LARGE_FILE)}")
    elif os.path.exists(SAMPLE_FILE):
        filepath = SAMPLE_FILE
        print(f"\nUsing sample dataset: {os.path.basename(SAMPLE_FILE)}")
    else:
        print("\n❌ No sample data files found!")
        print(f"   Expected: {SAMPLE_FILE}")
        print(f"   Or: {LARGE_FILE}")
        return 1
    
    # Run analysis pipeline
    df = load_and_inspect(filepath)
    
    if df is not None:
        analyze_missing_data(df)
        analyze_statistics(df)
        analyze_categories(df)
        analyze_outliers(df)
        analyze_correlations(df)
        generate_insights(df)
    
    print_header("✅ DEMO ANALYSIS COMPLETE")
    print("\n   To run the interactive dashboard:")
    print("   $ streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
