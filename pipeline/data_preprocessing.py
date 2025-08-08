"""
SA E-Commerce Customer Analytics - Data Preprocessing Pipeline
============================================================

Purpose: Comprehensive data preprocessing and feature engineering
Author: Aviwe Dlepu
Date: December 2024
Python Version: 3.8+

Dependencies:
- pandas>=1.5.3
- numpy>=1.24.3
- scikit-learn>=1.3.0
- google-cloud-bigquery>=3.11.4
- matplotlib>=3.7.1
- seaborn>=0.12.2
============================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from google.cloud import bigquery
from google.oauth2 import service_account
import os
from datetime import datetime, timedelta
import json

# Data preprocessing libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

class SAEcommerceDataProcessor:
    """
    Comprehensive data preprocessing pipeline for SA E-commerce Customer Analytics
    
    Features:
    - BigQuery integration
    - Data quality assessment
    - Feature engineering
    - Missing value handling
    - Outlier detection and treatment
    - Data transformation and scaling
    - Export capabilities
    """
    
    def __init__(self, project_id='customerinsightsavy', dataset_id='ecommerce_data', credentials_path=None):
        """
        Initialize the data processor
        
        Args:
            project_id (str): BigQuery project ID
            dataset_id (str): BigQuery dataset ID  
            credentials_path (str): Path to BigQuery service account key
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = None
        self.raw_data = {}
        self.processed_data = {}
        self.feature_engineered_data = None
        self.scalers = {}
        self.encoders = {}
        
        # Initialize BigQuery client
        self._initialize_bigquery_client(credentials_path)
        
        # Data quality metrics
        self.data_quality_report = {}
        
        print("🥑 SA E-Commerce Data Processor Initialized")
        print(f"📊 Project: {self.project_id}")
        print(f"🗃️ Dataset: {self.dataset_id}")
    
    def _initialize_bigquery_client(self, credentials_path):
        """Initialize BigQuery client with authentication"""
        try:
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = bigquery.Client(credentials=credentials, project=self.project_id)
                print("✅ BigQuery client initialized with service account")
            else:
                # Use application default credentials
                self.client = bigquery.Client(project=self.project_id)
                print("✅ BigQuery client initialized with default credentials")
        except Exception as e:
            print(f"❌ Error initializing BigQuery client: {e}")
            self.client = None
    
    def load_data_from_bigquery(self, use_sample=False, sample_size=10000):
        """
        Load data from BigQuery tables
        
        Args:
            use_sample (bool): Whether to use a sample of data
            sample_size (int): Size of sample if using sample data
        """
        print("📥 Loading data from BigQuery...")
        
        if not self.client:
            print("❌ BigQuery client not initialized")
            return False
        
        # Define table queries
        queries = {
            'customers': f"""
                SELECT * FROM `{self.project_id}.{self.dataset_id}.customers_dataset`
                {'TABLESAMPLE SYSTEM (10 PERCENT)' if use_sample else ''}
                {'LIMIT ' + str(sample_size) if use_sample else ''}
            """,
            'orders': f"""
                SELECT * FROM `{self.project_id}.{self.dataset_id}.order_history`
                {'TABLESAMPLE SYSTEM (10 PERCENT)' if use_sample else ''}
                {'LIMIT ' + str(sample_size) if use_sample else ''}
            """,
            'reviews': f"""
                SELECT * FROM `{self.project_id}.{self.dataset_id}.customer_reviews`
                {'TABLESAMPLE SYSTEM (10 PERCENT)' if use_sample else ''}
                {'LIMIT ' + str(sample_size) if use_sample else ''}
            """,
            'nps': f"""
                SELECT * FROM `{self.project_id}.{self.dataset_id}.nps_survey_data`
                {'TABLESAMPLE SYSTEM (10 PERCENT)' if use_sample else ''}
                {'LIMIT ' + str(sample_size) if use_sample else ''}
            """,
            'churn': f"""
                SELECT * FROM `{self.project_id}.{self.dataset_id}.customer_churn`
                {'TABLESAMPLE SYSTEM (10 PERCENT)' if use_sample else ''}
                {'LIMIT ' + str(sample_size) if use_sample else ''}
            """,
            'website_activity': f"""
                SELECT * FROM `{self.project_id}.{self.dataset_id}.website_activity_logs`
                {'TABLESAMPLE SYSTEM (10 PERCENT)' if use_sample else ''}
                {'LIMIT ' + str(sample_size) if use_sample else ''}
            """
        }
        
        # Load each table
        for table_name, query in queries.items():
            try:
                print(f"  📋 Loading {table_name}...")
                df = self.client.query(query).to_dataframe()
                self.raw_data[table_name] = df
                print(f"  ✅ {table_name}: {len(df):,} records loaded")
            except Exception as e:
                print(f"  ❌ Error loading {table_name}: {e}")
                return False
        
        print(f"🎉 Successfully loaded {len(self.raw_data)} tables")
        return True
    
    def load_data_from_csv(self, data_folder='data/'):
        """
        Load data from CSV files (alternative to BigQuery)
        
        Args:
            data_folder (str): Path to folder containing CSV files
        """
        print("📥 Loading data from CSV files...")
        
        csv_files = {
            'customers': 'customers_dataset.csv',
            'orders': 'order_history.csv', 
            'reviews': 'customer_reviews.csv',
            'nps': 'nps_survey_data.csv',
            'churn': 'customer_churn.csv',
            'website_activity': 'website_activity_logs.csv'
        }
        
        for table_name, filename in csv_files.items():
            filepath = os.path.join(data_folder, filename)
            try:
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    self.raw_data[table_name] = df
                    print(f"  ✅ {table_name}: {len(df):,} records loaded from {filename}")
                else:
                    print(f"  ⚠️ File not found: {filepath}")
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
        
        print(f"🎉 Successfully loaded {len(self.raw_data)} CSV files")
        return len(self.raw_data) > 0
    
    def assess_data_quality(self):
        """Comprehensive data quality assessment"""
        print("🔍 Assessing data quality...")
        
        quality_report = {}
        
        for table_name, df in self.raw_data.items():
            print(f"\n📊 Analyzing {table_name}:")
            
            table_report = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'duplicate_records': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Print summary
            print(f"  📈 Records: {table_report['total_records']:,}")
            print(f"  📊 Columns: {table_report['total_columns']}")
            print(f"  🔄 Duplicates: {table_report['duplicate_records']}")
            
            # Missing values analysis
            missing_cols = [col for col, pct in table_report['missing_percentage'].items() if pct > 0]
            if missing_cols:
                print(f"  ⚠️ Columns with missing values: {len(missing_cols)}")
                for col in missing_cols[:5]:  # Show top 5
                    pct = table_report['missing_percentage'][col]
                    print(f"    - {col}: {pct:.1f}%")
            else:
                print("  ✅ No missing values found")
            
            quality_report[table_name] = table_report
        
        self.data_quality_report = quality_report
        print("\n✅ Data quality assessment completed")
        return quality_report
    
    def clean_data(self):
        """Clean and standardize data across all tables"""
        print("🧹 Cleaning data...")
        
        for table_name, df in self.raw_data.items():
            print(f"  🔧 Cleaning {table_name}...")
            
            # Create a copy for cleaning
            cleaned_df = df.copy()
            
            # Remove duplicates
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            if len(cleaned_df) < initial_count:
                print(f"    🗑️ Removed {initial_count - len(cleaned_df)} duplicate records")
            
            # Handle missing values based on column type and importance
            if table_name == 'customers':
                cleaned_df = self._clean_customers_data(cleaned_df)
            elif table_name == 'orders':
                cleaned_df = self._clean_orders_data(cleaned_df)
            elif table_name == 'reviews':
                cleaned_df = self._clean_reviews_data(cleaned_df)
            elif table_name == 'nps':
                cleaned_df = self._clean_nps_data(cleaned_df)
            elif table_name == 'churn':
                cleaned_df = self._clean_churn_data(cleaned_df)
            elif table_name == 'website_activity':
                cleaned_df = self._clean_website_activity_data(cleaned_df)
            
            self.processed_data[table_name] = cleaned_df
            print(f"    ✅ {table_name} cleaned: {len(cleaned_df):,} records")
        
        print("🎉 Data cleaning completed")
        return True
    
    def _clean_customers_data(self, df):
        """Clean customer data"""
        # Handle missing ages with median by gender
        if 'Age' in df.columns and df['Age'].isnull().any():
            median_age_by_gender = df.groupby('Gender')['Age'].median()
            for gender in df['Gender'].unique():
                if pd.notna(gender):
                    mask = (df['Gender'] == gender) & df['Age'].isnull()
                    df.loc[mask, 'Age'] = median_age_by_gender[gender]
        
        # Fill remaining missing ages with overall median
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Clean gender data
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].fillna('Unknown')
            df['Gender'] = df['Gender'].str.title()
        
        # Clean province and city data
        if 'Province' in df.columns:
            df['Province'] = df['Province'].fillna('Unknown')
        
        if 'City' in df.columns:
            df['City'] = df['City'].fillna('Unknown')
        
        # Convert date columns
        date_columns = ['RegisteredDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Fill financial columns with 0
        financial_columns = ['TotalSpend', 'NumberOfOrders', 'NumberOfReturnedOrders', 'NumberOfCanceledOrders']
        for col in financial_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    
    def _clean_orders_data(self, df):
        """Clean order data"""
        # Convert date columns
        date_columns = ['OrderDate', 'PromisedDeliveryDate', 'ActualDeliveryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Fill missing numerical columns
        numerical_columns = ['Quantity', 'Price']
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical columns
        categorical_columns = ['OrderPlatform', 'PaymentMethod', 'ProductType', 'Category', 'ItemStatus']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _clean_reviews_data(self, df):
        """Clean review data"""
        # Remove records with missing review scores
        if 'ReviewScore' in df.columns:
            df = df.dropna(subset=['ReviewScore'])
        
        # Fill missing review text
        if 'ReviewText' in df.columns:
            df['ReviewText'] = df['ReviewText'].fillna('No review text provided')
        
        return df
    
    def _clean_nps_data(self, df):
        """Clean NPS data"""
        # Remove records with missing NPS scores
        if 'NPSScore' in df.columns:
            df = df.dropna(subset=['NPSScore'])
        
        # Convert survey completion date
        if 'SurveyCompletionDate' in df.columns:
            df['SurveyCompletionDate'] = pd.to_datetime(df['SurveyCompletionDate'], errors='coerce')
        
        # Fill missing categorical columns
        categorical_columns = ['SurveyChannel', 'NPSType', 'PurchaseFrequency', 'CustomerType']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _clean_churn_data(self, df):
        """Clean churn data"""
        # Fill missing categorical columns
        categorical_columns = ['ReasonForLeaving', 'OverallExperience', 'SpecificIssues', 'ChurnType']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _clean_website_activity_data(self, df):
        """Clean website activity data"""
        # Fill missing numerical columns with median
        numerical_columns = ['TimeSpentOnPages', 'PagesVisited']
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing session details
        if 'SessionDetails' in df.columns:
            df['SessionDetails'] = df['SessionDetails'].fillna('Unknown activity')
        
        return df
    
    def create_master_dataset(self):
        """Create master dataset by joining all tables"""
        print("🔗 Creating master dataset...")
        
        if 'customers' not in self.processed_data:
            print("❌ Customer data not available")
            return None
        
        # Start with customer data
        master_df = self.processed_data['customers'].copy()
        print(f"  📊 Base customers: {len(master_df):,}")
        
        # Add aggregated order data
        if 'orders' in self.processed_data:
            order_agg = self._aggregate_order_data()
            master_df = master_df.merge(order_agg, on='CustomerID', how='left')
            print(f"  📦 Added order metrics")
        
        # Add review aggregations
        if 'reviews' in self.processed_data:
            review_agg = self._aggregate_review_data()
            master_df = master_df.merge(review_agg, on='CustomerID', how='left')
            print(f"  ⭐ Added review metrics")
        
        # Add NPS data (latest score per customer)
        if 'nps' in self.processed_data:
            nps_latest = self._get_latest_nps_data()
            master_df = master_df.merge(nps_latest, on='CustomerID', how='left')
            print(f"  📈 Added NPS metrics")
        
        # Add churn data
        if 'churn' in self.processed_data:
            churn_data = self.processed_data['churn'][['CustomerID', 'ChurnType', 'OverallExperience']].copy()
            master_df = master_df.merge(churn_data, on='CustomerID', how='left')
            print(f"  🔄 Added churn metrics")
        
        # Add website activity aggregations
        if 'website_activity' in self.processed_data:
            activity_agg = self._aggregate_website_activity()
            master_df = master_df.merge(activity_agg, on='CustomerID', how='left')
            print(f"  🌐 Added website activity metrics")
        
        print(f"  ✅ Master dataset created: {len(master_df):,} customers with {len(master_df.columns)} features")
        return master_df
    
    def _aggregate_order_data(self):
        """Aggregate order data per customer"""
        orders_df = self.processed_data['orders'].copy()
        
        # Calculate order value
        orders_df['OrderValue'] = orders_df['Quantity'] * orders_df['Price']
        
        agg_dict = {
            'OrderID': 'count',
            'OrderValue': ['sum', 'mean', 'std'],
            'Quantity': ['sum', 'mean'],
            'Price': ['mean', 'max'],
            'OrderDate': ['min', 'max'],
            'Category': 'nunique',
            'ProductID': 'nunique',
            'OrderPlatform': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
        }
        
        order_agg = orders_df.groupby('CustomerID').agg(agg_dict)
        order_agg.columns = [
            'TotalOrders', 'TotalOrderValue', 'AvgOrderValue', 'OrderValueStd',
            'TotalQuantity', 'AvgQuantity', 'AvgPrice', 'MaxPrice',
            'FirstOrderDate', 'LastOrderDate', 'UniqueCategoriesPurchased',
            'UniqueProductsPurchased', 'PreferredPlatform'
        ]
        
        # Calculate recency
        order_agg['DaysSinceLastOrder'] = (pd.Timestamp.now() - order_agg['LastOrderDate']).dt.days
        order_agg['CustomerLifespanDays'] = (order_agg['LastOrderDate'] - order_agg['FirstOrderDate']).dt.days
        
        # Calculate frequency (orders per month)
        order_agg['OrderFrequency'] = order_agg['TotalOrders'] / (order_agg['CustomerLifespanDays'] / 30)
        order_agg['OrderFrequency'] = order_agg['OrderFrequency'].fillna(0)
        
        return order_agg.reset_index()
    
    def _aggregate_review_data(self):
        """Aggregate review data per customer"""
        reviews_df = self.processed_data['reviews'].copy()
        
        agg_dict = {
            'ReviewScore': ['count', 'mean', 'std', 'min', 'max'],
            'ReviewText': lambda x: x.str.len().mean()  # Average review length
        }
        
        review_agg = reviews_df.groupby('CustomerID').agg(agg_dict)
        review_agg.columns = [
            'TotalReviews', 'AvgReviewScore', 'ReviewScoreStd',
            'MinReviewScore', 'MaxReviewScore', 'AvgReviewLength'
        ]
        
        return review_agg.reset_index()
    
    def _get_latest_nps_data(self):
        """Get latest NPS data per customer"""
        nps_df = self.processed_data['nps'].copy()
        
        # Sort by survey completion date and get latest
        nps_df = nps_df.sort_values('SurveyCompletionDate', ascending=False)
        nps_latest = nps_df.groupby('CustomerID').first().reset_index()
        
        # Select relevant columns
        nps_columns = ['CustomerID', 'NPSScore', 'NPSType', 'PurchaseFrequency', 'CustomerType']
        nps_latest = nps_latest[nps_columns]
        
        return nps_latest
    
    def _aggregate_website_activity(self):
        """Aggregate website activity data per customer"""
        activity_df = self.processed_data['website_activity'].copy()
        
        agg_dict = {
            'TimeSpentOnPages': ['sum', 'mean', 'count'],
            'PagesVisited': ['sum', 'mean'],
            'SessionDetails': 'count'
        }
        
        activity_agg = activity_df.groupby('CustomerID').agg(agg_dict)
        activity_agg.columns = [
            'TotalTimeSpent', 'AvgTimePerSession', 'TotalSessions',
            'TotalPagesVisited', 'AvgPagesPerSession', 'SessionCount'
        ]
        
        # Calculate engagement score
        activity_agg['EngagementScore'] = (
            activity_agg['AvgTimePerSession'] * 0.4 + 
            activity_agg['AvgPagesPerSession'] * 0.6
        )
        
        return activity_agg.reset_index()
    
    def engineer_features(self, master_df):
        """Advanced feature engineering"""
        print("⚙️ Engineering features...")
        
        df = master_df.copy()
        
        # Age-based features
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], 
                                  bins=[0, 25, 35, 50, 100], 
                                  labels=['Young', 'Adult', 'Middle_Age', 'Senior'])
        
        # Tenure-based features
        if 'RegisteredDate' in df.columns:
            df['CustomerTenureDays'] = (pd.Timestamp.now() - df['RegisteredDate']).dt.days
            df['CustomerTenureMonths'] = df['CustomerTenureDays'] / 30
            df['TenureGroup'] = pd.cut(df['CustomerTenureMonths'],
                                     bins=[0, 3, 12, 24, 1000],
                                     labels=['New', 'Regular', 'Loyal', 'VIP'])
        
        # RFM Features
        if all(col in df.columns for col in ['DaysSinceLastOrder', 'TotalOrders', 'TotalSpend']):
            # Recency Score (lower is better)
            df['RecencyScore'] = pd.cut(df['DaysSinceLastOrder'], 
                                      bins=[0, 30, 90, 180, 1000], 
                                      labels=[5, 4, 3, 2, 1]).astype(float)
            
            # Frequency Score
            df['FrequencyScore'] = pd.qcut(df['TotalOrders'].fillna(0), 
                                         q=5, labels=[1, 2, 3, 4, 5]).astype(float)
            
            # Monetary Score
            df['MonetaryScore'] = pd.qcut(df['TotalSpend'].fillna(0), 
                                        q=5, labels=[1, 2, 3, 4, 5]).astype(float)
            
            # Combined RFM Score
            df['RFMScore'] = (df['RecencyScore'] + df['FrequencyScore'] + df['MonetaryScore']) / 3
        
        # Customer Lifetime Value (CLV)
        if all(col in df.columns for col in ['AvgOrderValue', 'OrderFrequency', 'CustomerTenureMonths']):
            # Simple CLV calculation
            df['CLV'] = (df['AvgOrderValue'].fillna(0) * 
                        df['OrderFrequency'].fillna(0) * 
                        df['CustomerTenureMonths'].fillna(1))
            
            # CLV segments
            df['CLVSegment'] = pd.qcut(df['CLV'].fillna(0), 
                                     q=4, labels=['Low', 'Medium', 'High', 'VIP'])
        
        # Churn risk features
        if 'ChurnType' in df.columns:
            df['ChurnRisk'] = df['ChurnType'].map({
                'Loyal': 0,
                'Engaged': 0.2,
                'Voluntary': 0.8,
                'Involuntary': 1.0
            }).fillna(0.5)
        
        # Customer value segmentation
        if all(col in df.columns for col in ['CLV', 'ChurnRisk']):
            # Create 2x2 matrix of CLV vs Churn Risk
            clv_median = df['CLV'].median()
            churn_median = df['ChurnRisk'].median()
            
            conditions = [
                (df['CLV'] >= clv_median) & (df['ChurnRisk'] <= churn_median),
                (df['CLV'] < clv_median) & (df['ChurnRisk'] <= churn_median),
                (df['CLV'] >= clv_median) & (df['ChurnRisk'] > churn_median),
                (df['CLV'] < clv_median) & (df['ChurnRisk'] > churn_median)
            ]
            
            choices = ['High_Value_Loyal', 'Low_Value_Loyal', 'High_Value_At_Risk', 'Low_Value_At_Risk']
            df['CustomerSegment'] = np.select(conditions, choices, default='Unknown')
        
        # Behavioral features
        if 'AvgReviewScore' in df.columns:
            df['ReviewSentiment'] = pd.cut(df['AvgReviewScore'].fillna(3), 
                                         bins=[0, 2, 3, 4, 5], 
                                         labels=['Negative', 'Neutral', 'Positive', 'Very_Positive'])
        
        if 'NPSScore' in df.columns:
            df['NPSCategory'] = pd.cut(df['NPSScore'].fillna(5), 
                                     bins=[0, 6, 8, 10], 
                                     labels=['Detractor', 'Passive', 'Promoter'])
        
        # Return rate calculation
        if all(col in df.columns for col in ['NumberOfReturnedOrders', 'NumberOfOrders']):
            df['ReturnRate'] = df['NumberOfReturnedOrders'] / df['NumberOfOrders'].replace(0, 1)
            df['ReturnRate'] = df['ReturnRate'].fillna(0)
            
            df['ReturnBehavior'] = pd.cut(df['ReturnRate'], 
                                        bins=[0, 0.1, 0.2, 1], 
                                        labels=['Low_Return', 'Medium_Return', 'High_Return'])
        
        # Geographic features
        if 'Province' in df.columns:
            # Major provinces
            major_provinces = ['Gauteng', 'Western Cape', 'KwaZulu-Natal']
            df['MajorProvince'] = df['Province'].isin(major_provinces)
        
        if 'City' in df.columns:
            # Major cities
            major_cities = ['Johannesburg', 'Cape Town', 'Durban', 'Pretoria']
            df['MajorCity'] = df['City'].isin(major_cities)
        
        # Engagement features
        if 'EngagementScore' in df.columns:
            df['EngagementLevel'] = pd.qcut(df['EngagementScore'].fillna(0), 
                                          q=3, labels=['Low', 'Medium', 'High'])
        
        print(f"  ✅ Feature engineering completed: {len(df.columns)} total features")
        self.feature_engineered_data = df
        return df
    
    def detect_and_handle_outliers(self, df, method='iqr', contamination=0.1):
        """Detect and handle outliers in numerical columns"""
        print(f"🔍 Detecting outliers using {method} method...")
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numerical_columns:
            if col in df.columns and not df[col].empty:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    
                    # Cap outliers at bounds
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                elif method == 'isolation_forest':
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    outlier_labels = iso_forest.fit_predict(df[[col]].fillna(df[col].median()))
                    outlier_count = sum(outlier_labels == -1)
                    
                    # Replace outliers with median
                    df.loc[outlier_labels == -1, col] = df[col].median()
                
                outlier_info[col] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': (outlier_count / len(df)) * 100
                }
                
                if outlier_count > 0:
                    print(f"  📊 {col}: {outlier_count} outliers ({outlier_info[col]['outlier_percentage']:.1f}%)")
        
        print(f"  ✅ Outlier detection completed for {len(numerical_columns)} numerical columns")
        return df, outlier_info
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables"""
        print("🔤 Encoding categorical variables...")
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        encoded_df = df.copy()
        
        for col in categorical_columns:
            if col in ['CustomerID']:  # Skip ID columns
                continue
                
            # Use label encoding for ordinal variables
            if col in ['AgeGroup', 'TenureGroup', 'CLVSegment', 'ReviewSentiment', 'EngagementLevel']:
                le = LabelEncoder()
                encoded_df[col + '_encoded'] = le.fit_transform(encoded_df[col].astype(str))
                self.encoders[col] = le
                print(f"  🏷️ Label encoded: {col}")
            
            # Use one-hot encoding for nominal variables
            else:
                # Get top categories to avoid too many dummy variables
                top_categories = encoded_df[col].value_counts().head(10).index
                encoded_df[col + '_top'] = encoded_df[col].where(encoded_df[col].isin(top_categories), 'Other')
                
                # Create dummy variables
                dummies = pd.get_dummies(encoded_df[col + '_top'], prefix=col, drop_first=True)
                encoded_df = pd.concat([encoded_df, dummies], axis=1)
                
                # Drop the temporary column
                encoded_df = encoded_df.drop(col + '_top', axis=1)
                print(f"  🎯 One-hot encoded: {col} ({len(dummies.columns)} features)")
        
        print(f"  ✅ Categorical encoding completed: {len(encoded_df.columns)} total features")
        return encoded_df
    
    def scale_numerical_features(self, df, scaling_method='standard'):
        """Scale numerical features"""
        print(f"📏 Scaling numerical features using {scaling_method} scaling...")
        
        # Select numerical columns (excluding encoded categoricals and IDs)
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        exclude_columns = ['CustomerID'] + [col for col in numerical_columns if col.endswith('_encoded')]
        
        scale_columns = [col for col in numerical_columns if col not in exclude_columns]
        
        scaled_df = df.copy()
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            print(f"❌ Unknown scaling method: {scaling_method}")
            return df
        
        if scale_columns:
            scaled_features = scaler.fit_transform(scaled_df[scale_columns].fillna(0))
            scaled_df[scale_columns] = scaled_features
            self.scalers[scaling_method] = scaler
            print(f"  ✅ Scaled {len(scale_columns)} numerical features")
        
        return scaled_df
    
    def create_train_test_split(self, df, target_column=None, test_size=0.2, random_state=42):
        """Create train-test split for modeling"""
        print(f"🔀 Creating train-test split ({test_size*100}% test)...")
        
        if target_column and target_column in df.columns:
            X = df.drop([target_column, 'CustomerID'], axis=1, errors='ignore')
            y = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"  ✅ Train set: {len(X_train):,} samples")
            print(f"  ✅ Test set: {len(X_test):,} samples")
            
            return X_train, X_test, y_train, y_test
        
        else:
            # Just split the data without target
            X = df.drop(['CustomerID'], axis=1, errors='ignore')
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            
            print(f"  ✅ Train set: {len(X_train):,} samples")
            print(f"  ✅ Test set: {len(X_test):,} samples")
            
            return X_train, X_test
    
    def save_processed_data(self, df, filename='processed_customer_data.csv', include_timestamp=True):
        """Save processed data to CSV"""
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename.split('.')[0]}_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"💾 Processed data saved to: {filename}")
        print(f"  📊 Shape: {df.shape}")
        return filename
    
    def generate_data_report(self):
        """Generate comprehensive data processing report"""
        print("📋 Generating data processing report...")
        
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'raw_data_summary': {},
            'processed_data_summary': {},
            'data_quality_report': self.data_quality_report,
            'feature_engineering_summary': {},
            'recommendations': []
        }
        
        # Raw data summary
        for table_name, df in self.raw_data.items():
            report['raw_data_summary'][table_name] = {
                'records': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().sum()
            }
        
        # Processed data summary
        if self.feature_engineered_data is not None:
            df = self.feature_engineered_data
            report['processed_data_summary'] = {
                'total_customers': len(df),
                'total_features': len(df.columns),
                'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns)
            }
        
        # Generate recommendations based on data quality
        recommendations = []
        
        # Check for high missing value rates
        for table_name, table_report in self.data_quality_report.items():
            high_missing_cols = [col for col, pct in table_report['missing_percentage'].items() if pct > 20]
            if high_missing_cols:
                recommendations.append(f"Consider additional data collection for {table_name}: {high_missing_cols}")
        
        # Check for low engagement
        if self.feature_engineered_data is not None and 'EngagementScore' in self.feature_engineered_data.columns:
            low_engagement_pct = (self.feature_engineered_data['EngagementScore'].fillna(0) < 1).mean() * 100
            if low_engagement_pct > 50:
                recommendations.append(f"High percentage of low-engagement customers ({low_engagement_pct:.1f}%) - consider engagement strategies")
        
        report['recommendations'] = recommendations
        
        # Save report as JSON
        report_filename = f"data_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Data processing report saved: {report_filename}")
        return report
    
    def run_full_pipeline(self, use_bigquery=True, use_sample=False, save_output=True):
        """Run the complete data preprocessing pipeline"""
        print("🚀 Starting SA E-Commerce Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        if use_bigquery:
            success = self.load_data_from_bigquery(use_sample=use_sample)
        else:
            success = self.load_data_from_csv()
        
        if not success:
            print("❌ Data loading failed. Pipeline terminated.")
            return None
        
        # Step 2: Assess data quality
        self.assess_data_quality()
        
        # Step 3: Clean data
        self.clean_data()
        
        # Step 4: Create master dataset
        master_df = self.create_master_dataset()
        if master_df is None:
            print("❌ Master dataset creation failed. Pipeline terminated.")
            return None
        
        # Step 5: Feature engineering
        final_df = self.engineer_features(master_df)
        
        # Step 6: Handle outliers
        final_df, outlier_info = self.detect_and_handle_outliers(final_df)
        
        # Step 7: Encode categorical variables
        final_df = self.encode_categorical_variables(final_df)
        
        # Step 8: Scale numerical features
        final_df = self.scale_numerical_features(final_df)
        
        # Step 9: Save processed data
        if save_output:
            output_filename = self.save_processed_data(final_df)
            self.generate_data_report()
        
        print("\n🎉 SA E-Commerce Data Preprocessing Pipeline Completed Successfully!")
        print(f"📊 Final dataset: {final_df.shape[0]:,} customers × {final_df.shape[1]} features")
        
        return final_df

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = SAEcommerceDataProcessor()
    
    # Run the full pipeline
    # For BigQuery access:
    # processed_data = processor.run_full_pipeline(use_bigquery=True, use_sample=False)
    
    # For CSV files:
    processed_data = processor.run_full_pipeline(use_bigquery=False, use_sample=False)
    
    if processed_data is not None:
        print("\n📈 Processing completed successfully!")
        print("🔍 Data ready for analysis and modeling")
    else:
        print("\n❌ Processing failed. Please check the logs.")
