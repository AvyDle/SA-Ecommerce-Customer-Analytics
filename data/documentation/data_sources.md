# 📊 **Data Sources Documentation**
## SA eCommerce Customer Analytics Project

*Comprehensive documentation of data collection, processing, and quality assurance*

---

## 🎯 **Project Data Overview**

**Project:** South African E-commerce Customer Analytics  
**Analysis Period:** January 2023 - September 2023  
**Geographic Scope:** All 9 South African Provinces  
**Customer Base:** 44,000+ active e-commerce customers  
**Data Architecture:** Google BigQuery Cloud Data Warehouse  

---

## 🏗️ **Data Architecture & Infrastructure**

### **📊 Source System Architecture**
```
E-commerce Platform (Operational Systems)
    ↓
Transactional Databases (MySQL/PostgreSQL)
    ↓
ETL Pipeline (Python + SQL)
    ↓
Google BigQuery Data Warehouse
    ↓
Analytics Layer (Jupyter + Tableau)
```

### **☁️ Cloud Infrastructure**
- **Primary Storage:** Google BigQuery
- **Project ID:** `customerinsightsavy`
- **Dataset ID:** `ecommerce_data`
- **Location:** United States (US)
- **Access Control:** IAM-based authentication
- **Data Encryption:** At-rest and in-transit encryption

---

## 📊 **Dataset Sources & Collection Methods**

### **1. 🏪 CUSTOMERS DATASET**

**Source:** Customer Relationship Management (CRM) System  
**Collection Method:** Automated data extraction from registration and profile systems  
**Update Frequency:** Daily incremental updates  
**Data Lineage:** Registration forms → CRM Database → BigQuery ETL → Analytics  

**📋 Collection Details:**
- **Registration Data:** Captured during account creation
- **Profile Updates:** Customer-initiated profile changes
- **Spending Calculations:** Aggregated from order history
- **Geographic Data:** Self-reported during registration
- **Data Validation:** Real-time validation during registration

**🔒 Privacy & Compliance:**
- Email addresses anonymized (customer0@example.com format)
- Phone numbers masked (last 3 digits removed)
- POPIA (Protection of Personal Information Act) compliant
- GDPR-aligned data handling procedures

---

### **2. 🛒 ORDER HISTORY DATASET**

**Source:** E-commerce Transaction System  
**Collection Method:** Real-time transaction logging  
**Update Frequency:** Real-time updates  
**Data Lineage:** Purchase → Payment Gateway → Order Management → BigQuery  

**📋 Collection Details:**
- **Transaction Capture:** Real-time order processing
- **Payment Integration:** Multiple payment gateways (Payfast, COD, Discovery Miles)
- **Inventory Integration:** Real-time stock status updates
- **Delivery Tracking:** Third-party logistics integration
- **Platform Detection:** Automatic device/platform identification

**🏪 Business Process Integration:**
- **Order Management:** Seamless integration with fulfillment systems
- **Inventory Updates:** Real-time stock level adjustments
- **Customer Notifications:** Automated order status communications
- **Return Processing:** Integrated return and refund tracking

---

### **3. ⭐ CUSTOMER REVIEWS DATASET**

**Source:** Product Review System  
**Collection Method:** Post-purchase review collection  
**Update Frequency:** Real-time review submissions  
**Data Lineage:** Customer Review → Moderation → Review Database → BigQuery  

**📋 Collection Details:**
- **Review Triggers:** Automated post-purchase email campaigns
- **Review Platform:** Integrated e-commerce review system
- **Moderation Process:** Automated spam filtering + manual review
- **Response Tracking:** Customer engagement with review prompts
- **Product Linking:** Automatic product-review association

**📈 Quality Assurance:**
- **Spam Filtering:** Automated detection of fake reviews
- **Content Moderation:** Review content guidelines enforcement
- **Verification:** Verified purchase requirement for reviews
- **Response Rates:** ~35% review completion rate

---

### **4. 🎯 NPS SURVEY DATA DATASET**

**Source:** Customer Experience Management System  
**Collection Method:** Multi-channel survey distribution  
**Update Frequency:** Monthly survey campaigns  
**Data Lineage:** Survey Distribution → Response Collection → Analysis Platform → BigQuery  

**📋 Collection Details:**
- **Survey Channels:** Website popups, Email campaigns, SMS notifications
- **Survey Timing:** Post-purchase (7-day delay), Quarterly relationship surveys
- **Response Tracking:** Multi-touch attribution for survey completion
- **Incentivization:** Optional discount codes for survey completion
- **Question Design:** Standard NPS methodology with custom follow-up questions

**📊 Survey Methodology:**
- **NPS Question:** "How likely are you to recommend us to a friend or colleague?"
- **Scale:** 0-10 (0 = Not at all likely, 10 = Extremely likely)
- **Classification:** Detractors (0-6), Passives (7-8), Promoters (9-10)
- **Follow-up Questions:** Open-ended feedback and specific improvement areas
- **Response Rate:** 18% average response rate across all channels

---

### **5. 🔄 CUSTOMER CHURN DATASET**

**Source:** Customer Lifecycle Management System  
**Collection Method:** Automated churn detection + exit surveys  
**Update Frequency:** Monthly churn evaluation  
**Data Lineage:** Customer Activity → Churn Algorithm → Exit Survey → BigQuery  

**📋 Collection Details:**
- **Churn Definition:** No purchase activity for 90+ days
- **Detection Method:** Automated analysis of purchase patterns
- **Exit Surveys:** Email-based surveys to recently churned customers
- **Winback Campaigns:** Targeted re-engagement efforts
- **Churn Scoring:** Predictive churn probability scoring

**🔍 Churn Classification Logic:**
- **Active:** Recent purchase (< 30 days)
- **Engaged:** Regular purchase pattern (30-60 days)
- **At Risk:** Declining activity (60-90 days)
- **Churned:** No activity (90+ days)
- **Voluntary vs Involuntary:** Self-reported vs system-detected

---

### **6. 🌐 WEBSITE ACTIVITY LOGS DATASET**

**Source:** Web Analytics Platform  
**Collection Method:** JavaScript tracking + server logs  
**Update Frequency:** Real-time data streaming  
**Data Lineage:** Website Interaction → Analytics Platform → Data Processing → BigQuery  

**📋 Collection Details:**
- **Tracking Method:** Google Analytics + Custom event tracking
- **Session Definition:** 30-minute inactivity timeout
- **Page Tracking:** All customer-facing pages monitored
- **Event Tracking:** Custom business events (add to cart, checkout, etc.)
- **Device Detection:** Automatic device and browser identification

**🔒 Privacy Compliance:**
- **Cookie Consent:** GDPR-compliant cookie management
- **IP Anonymization:** Last octet of IP addresses masked
- **Personal Data:** No PII collected in web analytics
- **Data Retention:** 26-month retention policy

---

## 📊 **Data Processing & ETL Pipeline**

### **🔄 ETL Process Flow**
```
1. EXTRACTION
   ├── Source Systems APIs
   ├── Database Exports
   └── Real-time Streaming

2. TRANSFORMATION
   ├── Data Cleaning
   ├── Format Standardization
   ├── Business Rule Application
   └── Quality Validation

3. LOADING
   ├── BigQuery Data Warehouse
   ├── Incremental Updates
   └── Full Refresh (Monthly)
```

### **⚙️ Processing Schedule**
- **Real-time:** Order transactions, website activity
- **Daily:** Customer data updates, review submissions
- **Weekly:** NPS survey data aggregation
- **Monthly:** Churn analysis, full data quality validation

### **🔍 Data Quality Controls**
- **Completeness Checks:** Missing value validation
- **Consistency Validation:** Cross-dataset relationship verification
- **Accuracy Testing:** Sample validation against source systems
- **Freshness Monitoring:** Data lag monitoring and alerting
- **Schema Evolution:** Automated schema change detection

---

## 📈 **Data Governance & Quality**

### **🎯 Data Quality Metrics**
| Dataset | Completeness | Accuracy | Timeliness | Consistency |
|---------|--------------|----------|------------|-------------|
| **Customers** | 98.5% | 99.2% | < 24 hours | 99.8% |
| **Orders** | 99.8% | 99.9% | Real-time | 99.9% |
| **Reviews** | 95.2% | 98.5% | Real-time | 98.8% |
| **NPS** | 89.3% | 99.1% | Weekly | 99.2% |
| **Churn** | 97.8% | 96.5% | Monthly | 98.5% |
| **Website** | 99.1% | 98.8% | Real-time | 99.3% |

### **🔒 Data Security & Compliance**
- **Access Control:** Role-based access to sensitive data
- **Encryption:** AES-256 encryption for data at rest
- **Audit Logging:** Complete audit trail for data access
- **Data Masking:** PII anonymization for analytics use
- **Compliance:** POPIA, GDPR, and industry standard compliance

### **📋 Data Retention Policies**
- **Transactional Data:** 7 years retention
- **Customer Data:** Active + 5 years post-churn
- **Analytics Data:** 3 years rolling retention
- **Web Logs:** 26 months (analytics standard)
- **Backup Data:** 90-day backup retention

---

## 🛠️ **Technical Implementation**

### **💻 Technology Stack**
- **Source Databases:** MySQL, PostgreSQL
- **ETL Framework:** Apache Airflow + Python
- **Data Warehouse:** Google BigQuery
- **Analytics Platform:** Jupyter Notebooks + Tableau
- **Monitoring:** Datadog + Custom alerting

### **📊 BigQuery Implementation Details**
```sql
-- Dataset Structure
Project: customerinsightsavy
├── Dataset: ecommerce_data
    ├── Table: Customers_Dataset
    ├── Table: Order_History
    ├── Table: Customer_Reviews
    ├── Table: Nps_Survey_Data
    ├── Table: Customer_Churn
    └── Table: Website_Activity_Logs
```

### **🔄 Performance Optimization**
- **Partitioning:** Date-based partitioning for large tables
- **Clustering:** Customer ID clustering for join optimization
- **Materialized Views:** Pre-aggregated metrics for fast reporting
- **Query Optimization:** Indexed columns and query pattern analysis

---

## 📊 **Data Limitations & Considerations**

### **⚠️ Known Limitations**
1. **Temporal Coverage:** 9-month analysis window (Jan-Sep 2023)
2. **Geographic Scope:** Limited to South African market
3. **Customer Representation:** E-commerce customers only (digital bias)
4. **Survey Response Bias:** Self-selected survey responses
5. **Churn Definition:** 90-day threshold may not suit all business models

### **🔍 Data Quality Considerations**
- **Seasonal Patterns:** Data may reflect seasonal e-commerce trends
- **Platform Bias:** Mobile vs desktop usage patterns may vary
- **Economic Factors:** 2023 economic conditions reflected in spending patterns
- **Sample Size:** 44,000 customers representative of target market segment

### **📈 Recommended Enhancements**
1. **Extended Time Series:** Multi-year historical data for trend analysis
2. **External Data Integration:** Economic indicators, competitor analysis
3. **Real-time Churn Scoring:** Live churn prediction model implementation
4. **Enhanced Attribution:** Multi-touch customer journey tracking
5. **Predictive Data Collection:** Forward-looking customer intent signals

---

## 📞 **Data Support & Contact**

**Data Engineering Team:**  
- **Lead:** Aviwe Dlepu
- **Email:** aviwedl@gmail.com
- **Documentation:** [GitHub Repository](https://github.com/AvyDle/SA-eCommerce-Customer-Analytics)

**BigQuery Access:**  
- **Project:** customerinsightsavy
- **Dataset:** ecommerce_data
- **Access:** IAM-controlled, request via project lead

**Support Documentation:**  
- **Setup Guide:** [Environment Setup Instructions]
- **Query Examples:** [Sample SQL Queries]
- **API Documentation:** [Data Access Patterns]

---

*Data Sources Documentation compiled by: Aviwe Dlepu*  
*Last Updated: September 2023*  
*SA eCommerce Customer Analytics Project*  
*Version: 1.0*