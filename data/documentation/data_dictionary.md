# 📊 **Data Dictionary**
## SA eCommerce Customer Analytics Project

*Comprehensive documentation of all datasets and variables used in the analysis*

---

## 📋 **Dataset Overview**

| Dataset | Records | Purpose | Key Variables |
|---------|---------|---------|---------------|
| **Customers Dataset** | 44,000 | Customer demographics and profiles | Age, Gender, Province, TotalSpend |
| **Order History** | 44,000 | Transaction and purchase behavior | OrderValue, OrderDate, Platform, Category |
| **Customer Reviews** | 44,000 | Product feedback and satisfaction | ReviewScore, ProductID, ReviewText |
| **NPS Survey Data** | 44,000 | Customer loyalty and advocacy | NPSScore, NPSType, SurveyChannel |
| **Customer Churn** | 44,000 | Customer retention patterns | ChurnType, ReasonForLeaving, Experience |
| **Website Activity** | 44,000 | Digital engagement metrics | TimeSpent, PagesVisited, SessionDetails |

---

## 🏗️ **1. CUSTOMERS DATASET**

**File:** `customers_dataset.csv`  
**Purpose:** Core customer demographics and spending profiles  
**Granularity:** One record per customer  

### **📊 Variable Definitions**

| Variable | Data Type | Description | Example Values | Notes |
|----------|-----------|-------------|----------------|-------|
| **CustomerID** | String | Unique customer identifier | CUST100001, CUST100002 | Primary key, format: CUST + 6 digits |
| **Age** | Integer | Customer age in years | 25, 34, 67 | Range: 18-100 years |
| **Gender** | String | Customer gender | Male, Female | Binary classification |
| **City** | String | Customer city location | Johannesburg, Cape Town | South African cities |
| **Province** | String | South African province | Gauteng, Western Cape | 9 SA provinces |
| **RegisteredDate** | Date | Account registration date | 5/16/2023, 2/12/2023 | Format: MM/DD/YYYY |
| **EmailAddress** | String | Customer email (anonymized) | customer0@example.com | Anonymized for privacy |
| **PhoneNumber** | String | Customer phone (anonymized) | 721030057 | Anonymized for privacy |
| **TotalSpend** | Float | Lifetime customer spending | 2303.77, 4434.22 | South African Rand (ZAR) |
| **NumberOfOrders** | Integer | Total orders placed | 89, 94, 56 | Count of all orders |
| **NumberOfReturnedOrders** | Integer | Orders returned | 35, 10, 27 | Count of returned orders |
| **NumberOfCanceledOrders** | Integer | Orders cancelled | 8, 83, 23 | Count of cancelled orders |

### **📈 Business Metrics**
- **Customer Lifetime Value (CLV):** Calculated from TotalSpend
- **Return Rate:** NumberOfReturnedOrders / NumberOfOrders
- **Cancellation Rate:** NumberOfCanceledOrders / NumberOfOrders
- **Average Order Value:** TotalSpend / NumberOfOrders

---

## 🛒 **2. ORDER HISTORY DATASET**

**File:** `order_history.csv`  
**Purpose:** Detailed transaction and purchase behavior analysis  
**Granularity:** One record per order  

### **📊 Variable Definitions**

| Variable | Data Type | Description | Example Values | Notes |
|----------|-----------|-------------|----------------|-------|
| **CustomerID** | String | Customer identifier (FK) | CUST102587 | Links to Customers table |
| **OrderID** | String | Unique order identifier | ORD100001, ORD100002 | Primary key |
| **ProductID** | String | Product identifier | PROD125273, PROD108142 | Links to product catalog |
| **Quantity** | Integer | Items ordered | 3, 5, 1 | Number of units |
| **Price** | Float | Order value | 274.395, 202.452 | South African Rand (ZAR) |
| **OrderDate** | Date | Order placement date | 7/10/2024, 11/26/2024 | Format: MM/DD/YYYY |
| **PromisedDeliveryDate** | Date | Expected delivery date | 7/17/2024, 12/5/2024 | Format: MM/DD/YYYY |
| **ActualDeliveryDate** | Date | Actual delivery date | 7/15/2024, 11/27/2024 | Format: MM/DD/YYYY |
| **OrderPlatform** | String | Purchase channel | Web, iOS, Android | Device/platform used |
| **Address** | String | Delivery address (anonymized) | Address2270, Address748 | Anonymized for privacy |
| **PaymentMethod** | String | Payment type | Payfast, COD, Discovery Miles | Payment gateway used |
| **ProductName** | String | Product description | Deluxe LED Headlight Bulbs | Product name |
| **ProductType** | String | Product classification | Retail | Product category type |
| **Category** | String | Product category | Auto and parts, Children toys | Business category |
| **ItemStatus** | String | Inventory status | Pre Order, In Stock | Stock availability |
| **SupplierName** | String | Product supplier | SupplierA, SupplierB | Vendor identification |
| **ShippingMethod** | String | Delivery method | Guaranteed Weekend, Next day | Shipping option |
| **DistributionCenter** | String | Fulfillment location | NW, EC, FS | Provincial code |
| **Channel** | String | Sales channel | Web, Android, iOS | Order channel |
| **FailedReason** | String | Delivery failure reason | No driver available | Failure description |
| **DeliveryStatus** | String | Final delivery status | Order Cancelled, Early | Delivery outcome |

---

## ⭐ **3. CUSTOMER REVIEWS DATASET**

**File:** `customer_reviews.csv`  
**Purpose:** Customer satisfaction and product feedback analysis  
**Granularity:** One record per review  

### **📊 Variable Definitions**

| Variable | Data Type | Description | Example Values | Notes |
|----------|-----------|-------------|----------------|-------|
| **CustomerID** | String | Customer identifier (FK) | CUST101922, CUST104307 | Links to Customers table |
| **ProductID** | String | Product identifier | PROD119578, PROD112456 | Product reviewed |
| **ReviewScore** | Integer | Rating given | 5, 2, 1, 3 | Scale: 1-5 (1=Poor, 5=Excellent) |
| **ReviewText** | String | Written feedback | "Amazing product! I love it." | Customer comment |

---

## 🎯 **4. NPS SURVEY DATA DATASET**

**File:** `nps_survey_data.csv`  
**Purpose:** Net Promoter Score and customer loyalty measurement  
**Granularity:** One record per survey response  

### **📊 Variable Definitions**

| Variable | Data Type | Description | Example Values | Notes |
|----------|-----------|-------------|----------------|-------|
| **CustomerID** | String | Customer identifier (FK) | CUST109441, CUST101811 | Links to Customers table |
| **SurveyCompletionDate** | Date | Survey completion date | 8/14/2024, 7/20/2024 | Format: MM/DD/YYYY |
| **SurveyChannel** | String | Survey distribution method | Website Popup, SMS, Email | Channel used |
| **NPSScore** | Integer | Net Promoter Score | 8, 7, 9, 4 | Scale: 0-10 |
| **NPSType** | String | NPS classification | Passive, Promoter, Detractor | Score categorization |
| **FollowUpResponse** | String | Additional feedback | "Improve customer support" | Open-ended response |
| **PurchaseFrequency** | String | Purchase behavior | Weekly, Rarely, Monthly | Frequency category |
| **CustomerType** | String | Customer classification | New, Returning | Customer status |
| **ProductFeedback** | String | Product-specific feedback | "Product range is okay" | Product comments |

---

## 🔄 **5. CUSTOMER CHURN DATASET**

**File:** `customer_churn.csv`  
**Purpose:** Customer retention patterns and churn analysis  
**Granularity:** One record per customer  

### **📊 Variable Definitions**

| Variable | Data Type | Description | Example Values | Notes |
|----------|-----------|-------------|----------------|-------|
| **CustomerID** | String | Customer identifier (FK) | CUST100001, CUST100002 | Links to Customers table |
| **ReasonForLeaving** | String | Churn reason | "Better options elsewhere", None | Why customer left |
| **OverallExperience** | String | Experience rating | Neutral, Dissatisfied, Satisfied | Overall satisfaction |
| **SpecificIssues** | String | Specific problems | "Payment issues", None | Detailed issues |
| **Suggestions** | String | Customer suggestions | "Better customer support" | Improvement ideas |
| **FollowUp** | String | Follow-up willingness | Yes, No | Re-engagement potential |
| **ChurnType** | String | Churn classification | Involuntary, Engaged, Loyal | Churn category |

---

## 🌐 **6. WEBSITE ACTIVITY LOGS DATASET**

**File:** `website_activity_logs.csv`  
**Purpose:** Digital engagement and website behavior analysis  
**Granularity:** One record per customer session summary  

### **📊 Variable Definitions**

| Variable | Data Type | Description | Example Values | Notes |
|----------|-----------|-------------|----------------|-------|
| **CustomerID** | String | Customer identifier (FK) | CUST118221, CUST131293 | Links to Customers table |
| **TimeSpentOnPages** | Float | Session duration | 4.75965, 11.4555 | Minutes spent on site |
| **PagesVisited** | Integer | Pages viewed | 17, 28, 26 | Number of pages visited |
| **SessionDetails** | String | Session outcome | "Checked out", "Subscribed" | Session result |

---

## 📊 **Data Quality Standards**

### **🔍 Data Validation Rules**
- **CustomerID:** Must be unique, format CUST + 6 digits
- **Dates:** Valid date format, no future dates for historical data
- **Numeric Fields:** Non-negative values where applicable
- **Geographic Data:** Valid South African provinces and cities
- **Score Fields:** Within specified ranges (e.g., NPS: 0-10, Reviews: 1-5)

### **📈 Data Completeness**
- **High Completeness (>95%):** CustomerID, Age, Gender, Province
- **Medium Completeness (80-95%):** Contact information, product details
- **Variable Completeness:** Some fields intentionally sparse (e.g., churn reasons for non-churned customers)

---

*Data Dictionary compiled by: Aviwe Dlepu*  
*Last Updated: September 2023*  
*SA eCommerce Customer Analytics Project*