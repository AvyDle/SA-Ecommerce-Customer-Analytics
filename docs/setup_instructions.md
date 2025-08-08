# Setup Instructions

## Overview

This document provides comprehensive setup instructions for reproducing the SA eCommerce Customer Analytics project environment. Follow these steps to configure your development environment and run the analysis notebooks and dashboards.

## 📋 Prerequisites

### System Requirements
- **Operating System:** Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **RAM:** Minimum 8GB (16GB recommended)
- **Storage:** 10GB free space
- **Internet Connection:** Required for data access and package downloads

### Required Software
- **Python:** Version 3.8 or higher
- **Git:** For version control
- **Web Browser:** Chrome, Firefox, Safari, or Edge (latest versions)

## 🐍 Python Environment Setup

### Step 1: Install Python
Download and install Python from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version
pip --version
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv sa_ecommerce_env

# Activate virtual environment
# On Windows:
sa_ecommerce_env\Scripts\activate

# On macOS/Linux:
source sa_ecommerce_env/bin/activate
```

### Step 3: Install Required Packages
Install all dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

If requirements.txt is not available, install packages manually:

```bash
# Core data science packages
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install plotly==5.15.0

# Machine learning
pip install scikit-learn==1.3.0
pip install xgboost==1.7.6
pip install imbalanced-learn==0.11.0

# Statistical analysis
pip install scipy==1.11.1
pip install statsmodels==0.14.0

# Google Cloud and BigQuery
pip install google-cloud-bigquery==3.11.4
pip install google-auth==2.22.0
pip install google-auth-oauthlib==1.0.0

# Data processing
pip install requests==2.31.0
pip install openpyxl==3.1.2

# Jupyter
pip install jupyter==1.0.0
pip install ipykernel==6.25.0
pip install ipywidgets==8.0.7

# Market basket analysis
pip install mlxtend==0.22.0

# Visualization enhancements
pip install wordcloud==1.9.2
pip install bokeh==3.2.1

# Utilities
pip install python-dotenv==1.0.0
pip install tqdm==4.65.0
```

## 🗃️ Project Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/sa-ecommerce-analytics.git
cd sa-ecommerce-analytics
```

### Step 2: Directory Structure
Ensure your project structure matches:
```
sa-ecommerce-analytics/
├── data/
│   ├── customer_churn.csv
│   ├── customer_reviews.csv
│   ├── customers_dataset.csv
│   ├── nps_survey_data.csv
│   ├── order_history.csv
│   └── website_activity_logs.csv
├── notebooks/
│   ├── eda_notebook.ipynb
│   ├── customer_segmentation.ipynb
│   ├── churn_modeling.ipynb
│   └── eci.ipynb
├── dashboards/
│   ├── screenshots/
│   ├── dashboard_links.md
│   └── story_points.md
├── docs/
│   ├── methodology.md
│   ├── business_insights.md
│   ├── model_documentation.md
│   └── setup_instructions.md
├── src/
│   ├── bigquery_etl.sql
│   └── data_preprocessing.py
├── requirements.txt
├── .gitignore
├── README.md
└── academic_report.docx
```

## 🔑 Authentication Setup

### Google Cloud BigQuery Access

#### Option 1: Service Account (Recommended)
1. **Create Service Account:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to IAM & Admin > Service Accounts
   - Create new service account
   - Download JSON key file

2. **Set Environment Variable:**
   ```bash
   # Windows
   set GOOGLE_APPLICATION_CREDENTIALS=path\to\your\keyfile.json
   
   # macOS/Linux
   export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/keyfile.json
   ```

3. **Store in Project:**
   ```bash
   # Create credentials directory (add to .gitignore)
   mkdir credentials
   cp path/to/keyfile.json credentials/bigquery_key.json
   ```

#### Option 2: Application Default Credentials
```bash
# Install gcloud CLI
# Windows: Download from https://cloud.google.com/sdk/docs/install-windows
# macOS: brew install google-cloud-sdk
# Ubuntu: apt-get install google-cloud-sdk

# Authenticate
gcloud auth application-default login
```

### Environment Variables
Create a `.env` file in the project root:

```bash
# .env file
GOOGLE_APPLICATION_CREDENTIALS=credentials/bigquery_key.json
BIGQUERY_PROJECT_ID=customerinsightsavy
BIGQUERY_DATASET_ID=ecommerce_data
TABLEAU_PUBLIC_USERNAME=your_tableau_username
```

## 📊 Data Setup

### Option 1: Use Provided CSV Files
The project includes sample CSV files in the `data/` directory. These can be used for local analysis without BigQuery access.

### Option 2: Connect to BigQuery
If you have access to the original BigQuery dataset:

1. **Verify Connection:**
   ```python
   from google.cloud import bigquery
   
   # Initialize client
   client = bigquery.Client()
   
   # Test query
   query = '''
   SELECT COUNT(*) as customer_count 
   FROM `customerinsightsavy.ecommerce_data.customers_dataset`
   LIMIT 1
   '''
   
   result = client.query(query).to_dataframe()
   print(f"Connected! Customer count: {result['customer_count'].iloc[0]}")
   ```

2. **Download Data Locally (Optional):**
   ```python
   # Run the data export script
   python src/data_preprocessing.py
   ```

## 🚀 Running the Analysis

### Step 1: Start Jupyter
```bash
# Navigate to project directory
cd sa-ecommerce-analytics

# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Step 2: Run Notebooks in Order

#### 1. Exploratory Data Analysis
```bash
# Open and run eda_notebook.ipynb
# This notebook performs initial data exploration
```

#### 2. Customer Segmentation
```bash
# Open and run customer_segmentation.ipynb
# This creates the CLV vs Churn segmentation
```

#### 3. Churn Modeling
```bash
# Open and run churn_modeling.ipynb
# This builds the churn prediction model
```

#### 4. E-Commerce Intelligence
```bash
# Open and run eci.ipynb
# This provides additional business insights
```

### Step 3: Verify Outputs
Each notebook should generate:
- Data visualizations
- Model artifacts
- Performance metrics
- Business insights

## 📈 Tableau Setup

### Installing Tableau Public
1. Download from [Tableau Public](https://public.tableau.com/)
2. Install following the setup wizard
3. Create Tableau Public account

### Dashboard Setup
1. **Open Tableau Files:**
   - Geographic Performance Dashboard (.twbx)
   - Customer Segmentation Dashboard (.twbx)
   - Product Performance Dashboard (.twbx)

2. **Data Connection:**
   - Connect to local CSV files in `data/` directory
   - Update data source paths if necessary

3. **Publishing:**
   - Sign in to Tableau Public
   - Publish dashboards to your profile
   - Update links in `dashboards/dashboard_links.md`

## 🔧 Troubleshooting

### Common Issues

#### Python Package Conflicts
```bash
# If you encounter package conflicts, create a fresh environment
conda create -n sa_ecommerce python=3.8
conda activate sa_ecommerce
pip install -r requirements.txt
```

#### BigQuery Authentication Errors
```bash
# Clear existing credentials
gcloud auth revoke
gcloud auth application-default revoke

# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

#### Jupyter Kernel Issues
```bash
# Install kernel in virtual environment
python -m ipykernel install --user --name=sa_ecommerce_env
```

#### Memory Issues
If you encounter memory errors:
1. **Reduce Data Size:**
   ```python
   # Sample data for testing
   df_sample = df.sample(n=10000, random_state=42)
   ```

2. **Optimize Memory Usage:**
   ```python
   # Use appropriate data types
   df['CustomerID'] = df['CustomerID'].astype('category')
   df['Age'] = df['Age'].astype('int8')
   ```

3. **Process in Chunks:**
   ```python
   # Process large datasets in chunks
   chunk_size = 1000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       process_chunk(chunk)
   ```

### File Path Issues
```bash
# Windows users: Use forward slashes or raw strings
data_path = r'C:\Users\username\sa-ecommerce-analytics\data\customers.csv'
# or
data_path = 'C:/Users/username/sa-ecommerce-analytics/data/customers.csv'

# macOS/Linux users: Use absolute paths if relative paths fail
data_path = '/home/username/sa-ecommerce-analytics/data/customers.csv'
```

### Package Import Errors
```python
# If specific packages can't be imported, install individually
!pip install package_name

# Restart kernel after installation
```

## 🧪 Testing the Setup

### Environment Test Script
Create and run this test script to verify your setup:

```python
# test_setup.py
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from google.cloud import bigquery
import jupyter

def test_environment():
    print("Testing SA eCommerce Analytics Environment...")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    
    # Test data loading
    try:
        df = pd.read_csv('data/customers_dataset.csv')
        print(f"✅ Successfully loaded customer data: {len(df)} records")
    except FileNotFoundError:
        print("❌ Customer data not found. Check data directory.")
    
    # Test BigQuery connection (optional)
    try:
        client = bigquery.Client()
        print("✅ BigQuery connection successful")
    except Exception as e:
        print(f"⚠️ BigQuery connection failed: {e}")
    
    print("Environment test complete!")

if __name__ == "__main__":
    test_environment()
```

Run the test:
```bash
python test_setup.py
```

## 📱 Additional Tools

### Visual Studio Code Setup (Optional)
1. Install [VS Code](https://code.visualstudio.com/)
2. Install Python extension
3. Install Jupyter extension
4. Configure Python interpreter to use virtual environment

### Git Configuration
```bash
# Configure Git (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Clone with SSH (alternative)
git clone git@github.com:your-username/sa-ecommerce-analytics.git
```

## 🔄 Updates and Maintenance

### Keeping Dependencies Updated
```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package_name

# Update all packages (use with caution)
pip install --upgrade -r requirements.txt
```

### Environment Backup
```bash
# Create new requirements file
pip freeze > requirements_backup.txt

# Document environment
conda env export > environment.yml  # If using conda
```

## 📞 Support

### Getting Help
1. **Documentation:** Check project documentation in `docs/` folder
2. **Issues:** Create GitHub issue for project-specific problems
3. **Community:** Stack Overflow for general Python/data science questions

### Contact Information
- **Project Repository:** [GitHub Link]
- **Author:** [Your Name]
- **Email:** [Your Email]

---

**Setup Version:** 1.0  
**Last Updated:** December 2024  
**Compatibility:** Python 3.8+, Windows/macOS/Linux  
**Estimated Setup Time:** 30-60 minutes
