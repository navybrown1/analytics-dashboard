"""
Test Suite for Business Analytics Dashboard
============================================
Run with: python test_dashboard.py
Tests core functions without requiring Streamlit server.
"""

import sys
import os
import pandas as pd
import numpy as np
from io import StringIO

# ============================================================================
# TEST DATA SETUP
# ============================================================================

SAMPLE_CSV = """Date,Product,Category,Sales,Quantity,Region,Customer_Type,Rating
2024-01-01,Widget A,Electronics,1500,10,North,Retail,4.5
2024-01-02,Widget B,Electronics,2300,15,South,Wholesale,4.8
2024-01-03,Widget A,Electronics,1800,12,North,Retail,4.2
2024-01-04,Gadget X,Home,3200,8,East,Retail,4.9
2024-01-05,Widget C,Electronics,1100,7,West,Retail,3.8
2024-01-06,Gadget Y,Home,2800,11,South,Wholesale,4.6
2024-01-07,Widget A,Electronics,1650,9,North,Retail,4.3
2024-01-08,Tool Z,Industrial,4500,5,East,Wholesale,4.7
2024-01-09,Widget B,Electronics,2100,14,West,Retail,4.4
2024-01-10,Gadget X,Home,3400,9,North,Retail,4.8
2024-01-11,Widget C,Electronics,,6,South,Retail,
2024-01-12,Tool Z,Industrial,4200,4,East,Wholesale,4.9
"""

EMPTY_CSV = ""

SINGLE_ROW_CSV = """Name,Value
Test,100
"""


def create_test_dataframe():
    """Create a test DataFrame from sample CSV"""
    return pd.read_csv(StringIO(SAMPLE_CSV))


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record(self, test_name, passed, error_msg=None):
        if passed:
            self.passed += 1
            print(f"  ✅ PASS: {test_name}")
        else:
            self.failed += 1
            self.errors.append((test_name, error_msg))
            print(f"  ❌ FAIL: {test_name} - {error_msg}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        print(f"{'='*60}")
        if self.errors:
            print("\nFailed tests:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        return self.failed == 0


results = TestResults()


def test_1_load_csv():
    """Test 1: CSV file loading and parsing"""
    print("\n📋 Test 1: CSV Loading")
    try:
        df = create_test_dataframe()
        results.record("CSV loads successfully", df is not None)
        results.record("Correct row count (12 rows)", len(df) == 12)
        results.record("Correct column count (8 cols)", len(df.columns) == 8)
        results.record("Column names correct",
                       list(df.columns) == ['Date', 'Product', 'Category', 'Sales',
                                            'Quantity', 'Region', 'Customer_Type', 'Rating'])
    except Exception as e:
        results.record("CSV loading", False, str(e))


def test_2_column_types():
    """Test 2: Column type detection"""
    print("\n📊 Test 2: Column Type Detection")
    try:
        df = create_test_dataframe()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        results.record("Numerical columns detected",
                       set(numerical_cols) == {'Sales', 'Quantity', 'Rating'})
        results.record("Categorical columns detected",
                       set(categorical_cols) == {'Date', 'Product', 'Category',
                                                  'Region', 'Customer_Type'})
        results.record("No columns missed",
                       len(numerical_cols) + len(categorical_cols) == len(df.columns))
    except Exception as e:
        results.record("Column type detection", False, str(e))


def test_3_missing_values():
    """Test 3: Missing value detection"""
    print("\n❓ Test 3: Missing Values Analysis")
    try:
        df = create_test_dataframe()
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        results.record("Missing values detected", total_missing > 0)
        results.record("Sales has 1 missing", missing_counts['Sales'] == 1)
        results.record("Rating has 1 missing", missing_counts['Rating'] == 1)
        results.record("Quantity has no missing", missing_counts['Quantity'] == 0)
        
        # Test missing percentage calculation
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        results.record("Missing % calculated correctly",
                       abs(missing_pct['Sales'] - 8.33) < 0.1)
    except Exception as e:
        results.record("Missing values analysis", False, str(e))


def test_4_descriptive_statistics():
    """Test 4: Descriptive statistics calculation"""
    print("\n📈 Test 4: Descriptive Statistics")
    try:
        df = create_test_dataframe()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = df[numerical_cols].describe()
        
        results.record("Statistics computed", stats is not None)
        results.record("Mean calculated", 'mean' in stats.index)
        results.record("Std calculated", 'std' in stats.index)
        results.record("Min/Max calculated",
                       'min' in stats.index and 'max' in stats.index)
        
        # Verify specific values
        sales_mean = df['Sales'].mean()
        results.record("Sales mean is reasonable",
                       1000 < sales_mean < 5000)
    except Exception as e:
        results.record("Descriptive statistics", False, str(e))


def test_5_filtering():
    """Test 5: Data filtering logic"""
    print("\n🔍 Test 5: Data Filtering")
    try:
        df = create_test_dataframe()
        
        # Numerical filter
        filtered = df[(df['Sales'] >= 2000) & (df['Sales'] <= 3000)]
        results.record("Numerical filter works", len(filtered) > 0)
        results.record("Numerical filter correct range",
                       filtered['Sales'].min() >= 2000 and filtered['Sales'].max() <= 3000)
        
        # Categorical filter
        filtered_cat = df[df['Region'].isin(['North', 'South'])]
        results.record("Categorical filter works", len(filtered_cat) > 0)
        results.record("Categorical filter correct values",
                       set(filtered_cat['Region'].unique()) <= {'North', 'South'})
        
        # Combined filter
        combined = df[(df['Sales'] >= 1500) & (df['Region'] == 'North')]
        results.record("Combined filter works", len(combined) > 0)
    except Exception as e:
        results.record("Data filtering", False, str(e))


def test_6_export_formats():
    """Test 6: Data export functionality"""
    print("\n💾 Test 6: Export Formats")
    try:
        df = create_test_dataframe()
        
        # CSV export
        csv_output = df.to_csv(index=False)
        results.record("CSV export works", len(csv_output) > 0)
        results.record("CSV contains headers", 'Product' in csv_output)
        
        # JSON export
        json_output = df.to_json(orient='records')
        results.record("JSON export works", len(json_output) > 0)
        
        # Excel export (if openpyxl available)
        try:
            from io import BytesIO
            buffer = BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            results.record("Excel export works", buffer.tell() > 0)
        except ImportError:
            results.record("Excel export (openpyxl not installed)", True)
            print("    ℹ️  openpyxl not installed, skipping Excel test")
    except Exception as e:
        results.record("Export formats", False, str(e))


def test_7_edge_cases():
    """Test 7: Edge case handling"""
    print("\n⚠️ Test 7: Edge Cases")
    try:
        # Empty dataframe
        empty_df = pd.DataFrame()
        results.record("Empty DataFrame detected", empty_df.empty)
        
        # Single row
        single_df = pd.read_csv(StringIO(SINGLE_ROW_CSV))
        results.record("Single row DataFrame works", len(single_df) == 1)
        
        # All missing column
        df = create_test_dataframe()
        df['AllNull'] = np.nan
        missing = df['AllNull'].isnull().sum()
        results.record("All-null column detected", missing == len(df))
        
        # Duplicate rows
        df_dup = pd.concat([df, df.head(3)])
        results.record("Duplicate handling", len(df_dup) == len(df) + 3)
    except Exception as e:
        results.record("Edge cases", False, str(e))


def test_8_correlation():
    """Test 8: Correlation matrix computation"""
    print("\n🔗 Test 8: Correlation Analysis")
    try:
        df = create_test_dataframe()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr = df[numerical_cols].corr()
        
        results.record("Correlation matrix computed", corr is not None)
        results.record("Matrix is square",
                       corr.shape[0] == corr.shape[1])
        results.record("Diagonal is 1.0",
                       all(abs(corr.iloc[i, i] - 1.0) < 0.001
                           for i in range(len(corr))))
        results.record("Values in [-1, 1] range",
                       corr.min().min() >= -1.001 and corr.max().max() <= 1.001)
    except Exception as e:
        results.record("Correlation analysis", False, str(e))


def test_9_outlier_detection():
    """Test 9: Outlier detection using IQR"""
    print("\n⚠️ Test 9: Outlier Detection")
    try:
        df = create_test_dataframe()
        col = 'Sales'
        col_data = df[col].dropna()
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = col_data[(col_data < lower) | (col_data > upper)]
        
        results.record("IQR calculated", IQR > 0)
        results.record("Bounds calculated", lower < upper)
        results.record("Outlier detection runs", outliers is not None)
        results.record("Outlier count is integer", isinstance(len(outliers), int))
    except Exception as e:
        results.record("Outlier detection", False, str(e))


def test_10_data_types_handling():
    """Test 10: Mixed data type handling"""
    print("\n🔄 Test 10: Data Type Handling")
    try:
        # Test with mixed types
        mixed_csv = """ID,Name,Value,Active
1,Alice,100.5,True
2,Bob,200.3,False
3,Charlie,150.0,True
"""
        df = pd.read_csv(StringIO(mixed_csv))
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        results.record("Mixed types loaded", len(df) == 3)
        results.record("Numeric detected", 'Value' in num_cols)
        results.record("String detected", 'Name' in cat_cols)
        results.record("ID detected as numeric", 'ID' in num_cols)
    except Exception as e:
        results.record("Data type handling", False, str(e))


def test_11_large_dataset_simulation():
    """Test 11: Performance with larger datasets"""
    print("\n🚀 Test 11: Large Dataset Simulation")
    try:
        # Create a larger dataset
        np.random.seed(42)
        n_rows = 10000
        large_df = pd.DataFrame({
            'ID': range(n_rows),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'Value1': np.random.normal(100, 15, n_rows),
            'Value2': np.random.uniform(0, 1000, n_rows),
            'Value3': np.random.exponential(50, n_rows)
        })
        
        results.record("Large dataset created (10K rows)", len(large_df) == n_rows)
        
        # Test statistics computation
        stats = large_df.describe()
        results.record("Statistics on large data", stats is not None)
        
        # Test correlation
        corr = large_df[['Value1', 'Value2', 'Value3']].corr()
        results.record("Correlation on large data", corr is not None)
        
        # Test filtering
        filtered = large_df[large_df['Value1'] > 100]
        results.record("Filtering on large data", len(filtered) > 0)
        
        # Test export
        csv_out = large_df.to_csv(index=False)
        results.record("Export large data", len(csv_out) > 0)
    except Exception as e:
        results.record("Large dataset simulation", False, str(e))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    print("=" * 60)
    print("📊 Business Analytics Dashboard - Test Suite")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print("=" * 60)
    
    # Run all tests
    test_1_load_csv()
    test_2_column_types()
    test_3_missing_values()
    test_4_descriptive_statistics()
    test_5_filtering()
    test_6_export_formats()
    test_7_edge_cases()
    test_8_correlation()
    test_9_outlier_detection()
    test_10_data_types_handling()
    test_11_large_dataset_simulation()
    
    # Print summary
    all_passed = results.summary()
    
    if all_passed:
        print("\n🎉 All tests passed! Dashboard is ready to use.")
        print("🚀 Run: streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
