# Business Analytics Dashboard

A powerful, interactive Streamlit web application for analyzing business data with automated insights, dynamic filtering, and beautiful visualizations.

## Features

- **Data Upload & Processing**: Universal CSV support with automatic type detection
- **Summary Statistics**: Comprehensive descriptive stats, skewness, kurtosis
- **Smart Filtering**: Interactive sidebar controls for numerical, categorical, and date columns
- **7 Visualization Types**: Histogram, Bar Chart, Scatter Plot, Box Plot, Pie Chart, Line Chart, Grouped Analysis
- **Correlation Analysis**: Heatmap with pairwise correlation strengths
- **AI-Driven Insights**: Automated data quality scoring, pattern detection, outlier alerts
- **Outlier Detection**: IQR-based anomaly identification
- **Multi-Format Export**: Download filtered data as CSV, Excel, or JSON
- **Color Themes**: 6 built-in color themes
- **Performance Caching**: Cached computations for responsive analysis

## Quick Start

```bash
# Navigate to the project folder
cd analytics-dashboard

# Install dependencies
pip install -r requirements_enhanced.txt

# Run the enhanced dashboard
streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502
```

The dashboard will open automatically at `http://127.0.0.1:8502`.

> **Note:** This project includes a `.streamlit/config.toml` that defaults to
> port 8502 on 127.0.0.1, so a bare `streamlit run app_enhanced.py` from this
> directory also works. If a global `~/.streamlit/config.toml` sets a different
> port, the command-line flags above will take priority.

## Project Structure

```
analytics-dashboard/
├── .streamlit/config.toml  # Streamlit config (port 8502, 127.0.0.1)
├── app.py                  # Basic dashboard (simpler version)
├── app_enhanced.py         # Enhanced dashboard (full features)
├── requirements.txt        # Basic dependencies
├── requirements_enhanced.txt # Full dependencies (includes scipy)
├── sample_data.csv         # Sample dataset (30 rows)
├── sample_data_large.csv   # Larger sample dataset (70 rows)
├── test_dashboard.py       # Test suite (11 test cases)
├── demo_analysis.py        # CLI demo analysis script
├── README.md               # This file
├── QUICKSTART.md           # Quick start guide
├── FEATURES.md             # Features checklist
├── ENHANCEMENTS.md         # Enhancement comparison
├── DEPLOYMENT.md           # Deployment guide
└── PROJECT_SUMMARY.md      # Project overview
```

## Usage

### Upload Your Data

1. Launch the dashboard with `streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502`
2. Use the sidebar to upload any CSV file
3. The dashboard auto-detects column types and generates analysis

### Explore Tabs

| Tab | Description |
|-----|-------------|
| **Data Overview** | Column info, missing values chart, data preview |
| **Statistics** | Descriptive stats table, outlier detection, box plots |
| **Visualizations** | 7 chart types with interactive controls |
| **Correlations** | Heatmap, pairwise strengths, scatter explorer |
| **Insights** | AI-driven findings, data profiling, quality metrics |
| **Export** | Download filtered data in CSV, Excel, or JSON |

### Filter Data

- Use the sidebar to filter by numerical ranges, categorical values, or date ranges
- All charts and statistics update in real-time based on active filters

## Testing

```bash
# Run the test suite
python test_dashboard.py

# Run the demo analysis (CLI)
python demo_analysis.py
```

## Two Dashboard Versions

| Feature | `app.py` (Basic) | `app_enhanced.py` (Full) |
|---------|:-:|:-:|
| CSV Upload | Yes | Yes |
| Basic Statistics | Yes | Yes |
| Missing Values | Yes | Yes |
| Histogram + Bar Chart | Yes | Yes |
| 7 Visualization Types | - | Yes |
| Correlation Heatmap | - | Yes |
| Outlier Detection | - | Yes |
| AI Insights | Basic | Advanced |
| Multi-Format Export | CSV only | CSV + Excel + JSON |
| Color Themes | - | 6 themes |
| Caching | - | Yes |
| Date Filters | - | Yes |
| Tabbed Interface | - | Yes |

## Requirements

- Python 3.9+
- streamlit 1.31.0
- pandas 2.1.4
- plotly 5.18.0
- numpy 1.26.3
- openpyxl 3.1.2
- scipy 1.11.4 (enhanced version only)

## License

This project is provided as-is for educational and business use.
