# Mac M4 Setup Guide

Step-by-step instructions for setting up the Business Analytics Dashboard on a Mac with Apple Silicon (M1/M2/M3/M4).

## Prerequisites

### 1. Check Python Installation

```bash
python3 --version
```

If Python is not installed, install it via Homebrew:

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
```

### 2. Verify pip

```bash
pip3 --version
```

## Setup Steps

### Step 1: Navigate to the Project

```bash
cd ~/analytics-dashboard
```

### Step 2: Create a Virtual Environment (Recommended)

Using a virtual environment keeps your system Python clean:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### Step 3: Install Dependencies

```bash
# For the enhanced dashboard (recommended)
pip install -r requirements_enhanced.txt

# Or for the basic dashboard only
pip install -r requirements.txt
```

**Note for Apple Silicon:** All listed packages have native ARM64 wheels. If you encounter any issues:

```bash
# Force reinstall with no cache
pip install --no-cache-dir -r requirements_enhanced.txt
```

### Step 4: Run the Dashboard

```bash
# Enhanced version (recommended)
streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502

# Basic version
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

The dashboard will automatically open in your default browser at `http://127.0.0.1:8502`.

> **Note:** The project includes `.streamlit/config.toml` with these defaults,
> so a bare `streamlit run app_enhanced.py` from this directory also picks up
> port 8502. The flags above guarantee the correct port even if a global
> `~/.streamlit/config.toml` overrides the project config.

### Step 5: Test Everything

```bash
# Run the test suite
python test_dashboard.py

# Run the demo analysis
python demo_analysis.py
```

## Apple Silicon Specific Notes

### Performance

Apple Silicon Macs handle this dashboard very well:
- **Data loading**: Near-instant for files under 100MB
- **Visualizations**: Smooth Plotly rendering
- **Caching**: Takes advantage of fast unified memory

### Potential Issues & Fixes

**Issue: "externally-managed-environment" error**
```bash
# Use a virtual environment (recommended) or:
pip install --break-system-packages -r requirements_enhanced.txt
```

**Issue: scipy fails to install**
```bash
# Install with Homebrew's Python
brew install python@3.11
python3.11 -m pip install -r requirements_enhanced.txt
```

**Issue: Port 8502 already in use**
```bash
# Use a different port
streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8503
```

**Issue: "streamlit: command not found"**
```bash
# Run as a module
python3 -m streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502

# Or add to PATH
export PATH="$HOME/Library/Python/3.11/bin:$PATH"
```

## Deactivate Virtual Environment

When you're done working:

```bash
deactivate
```

## Daily Workflow

```bash
# 1. Open terminal
cd ~/analytics-dashboard

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run dashboard
streamlit run app_enhanced.py --server.address 127.0.0.1 --server.port 8502

# 4. When done
# Press Ctrl+C in terminal to stop
deactivate
```

## Updating

To update packages in the future:

```bash
source venv/bin/activate
pip install --upgrade -r requirements_enhanced.txt
```

## Uninstalling

```bash
# Remove virtual environment
rm -rf ~/analytics-dashboard/venv

# Or remove the entire project
rm -rf ~/analytics-dashboard
```
