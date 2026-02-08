# Quick Start Guide

Get the Business Analytics Dashboard running in under 2 minutes.

## Prerequisites

- Python 3.9 or higher installed
- pip package manager

## Step 1: Install Dependencies

```bash
cd analytics-dashboard
pip install -r requirements_enhanced.txt
```

## Step 2: Run the Dashboard

**Enhanced version (recommended):**
```bash
streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502
```

**Basic version:**
```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

> **Note:** The project includes `.streamlit/config.toml` with port 8502 and
> address 127.0.0.1, so a bare `streamlit run app_enhanced.py` from this
> directory also works. The flags above make it explicit.

## Step 3: Upload Data

1. The dashboard opens at `http://127.0.0.1:8502`
2. Click **"Browse files"** in the sidebar
3. Upload any CSV file (or use the included `sample_data.csv`)

## Step 4: Explore

- **Data Overview tab**: See column info and missing values
- **Statistics tab**: View descriptive stats and outliers
- **Visualizations tab**: Create 7 types of interactive charts
- **Correlations tab**: Explore variable relationships
- **Insights tab**: Read AI-generated findings
- **Export tab**: Download your filtered data

## Quick Test

```bash
# Run tests to verify everything works
python test_dashboard.py

# Run a CLI demo analysis
python demo_analysis.py
```

## Sample Data

Two sample files are included:

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `sample_data.csv` | 30 | 8 | Basic business data |
| `sample_data_large.csv` | 70 | 11 | Extended with Revenue, Cost, Profit |

## Troubleshooting

**"streamlit: command not found"**
```bash
pip install streamlit
# Or use: python -m streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502
```

**Port already in use**
```bash
# This project defaults to 8502. Pick another port if that's also taken:
streamlit run app_enhanced.py --server.port 8503
```

**Slow on large files**
- The enhanced version uses caching for better performance
- Consider filtering to reduce data volume
- Files under 50MB work best

## Next Steps

- Read `FEATURES.md` for the full feature list
- Read `DEPLOYMENT.md` for production deployment
- Read `PROJECT_SUMMARY.md` for the complete overview
