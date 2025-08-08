-- ===============================================================
-- SA E-Commerce Customer Analytics - BigQuery ETL Pipeline
-- ===============================================================
-- Purpose: Extract, Transform, and Load data for customer analytics
-- Author: Aviwe Dlepu
-- Date: December 2024
-- BigQuery Project: customerinsightsavy.ecommerce_data
-- ===============================================================

-- ===============================================================
-- 1. CUSTOMER DATA EXTRACTION WITH FEATURE ENGINEERING
-- ===============================================================

-- Extract customer data with calculated features
CREATE OR REPLACE VIEW `customerinsightsavy.ecommerce_data.customer_features` AS
SELECT 
    c.CustomerID,
    c.Age,
    c.Gender,
    c.City,
    c.Province,
    c.RegisteredDate,
    c.EmailAddress,
    c.PhoneNumber,
    c.TotalSpend,
    c.NumberOfOrders,
    c.NumberOfReturnedOrders,
    c.NumberOfCanceledOrders,
    
    -- Feature Engineering
    CASE 
        WHEN c.Age BETWEEN 18 AND 25 THEN 'Gen_Z'
        WHEN c.Age BETWEEN 26 AND 35 THEN 'Millennial'
        WHEN c.Age BETWEEN 36 AND 50 THEN 'Gen_X'
        WHEN c.Age > 50 THEN 'Boomer'
        ELSE 'Unknown'
    END AS Age_Group,
    
    -- Customer tenure in days
    DATE_DIFF(CURRENT_DATE(), DATE(c.RegisteredDate), DAY) AS Customer_Tenure_Days,
    
    -- Average order value
    CASE 
        WHEN c.NumberOfOrders > 0 THEN c.TotalSpend / c.NumberOfOrders
        ELSE 0
    END AS Avg_Order_Value,
    
    -- Return rate
    CASE 
        WHEN c.NumberOfOrders > 0 THEN c.NumberOfReturnedOrders / c.NumberOfOrders
        ELSE 0
    END AS Return_Rate,
    
    -- Cancellation rate
    CASE 
        WHEN c.NumberOfOrders > 0 THEN c.NumberOfCanceledOrders / c.NumberOfOrders
        ELSE 0
    END AS Cancellation_Rate,
    
    -- Customer lifetime value (simple calculation)
    c.TotalSpend AS CLV_Current,
    
    -- Urban vs Rural classification
    CASE 
        WHEN c.City IN ('Johannesburg', 'Cape Town', 'Durban', 'Pretoria', 'Port Elizabeth', 'Bloemfontein') 
        THEN 'Urban'
        ELSE 'Rural'
    END AS Location_Type,
    
    -- Province grouping
    CASE 
        WHEN c.Province IN ('Gauteng', 'Western Cape') THEN 'High_Performance'
        WHEN c.Province IN ('KwaZulu-Natal', 'Eastern Cape') THEN 'Medium_Performance'
        ELSE 'Low_Performance'
    END AS Province_Performance_Group

FROM `customerinsightsavy.ecommerce_data.customers_dataset` c;

-- ===============================================================
-- 2. ORDER ANALYSIS WITH AGGREGATIONS
-- ===============================================================

-- Create order analysis view with product and delivery insights
CREATE OR REPLACE VIEW `customerinsightsavy.ecommerce_data.order_analysis` AS
SELECT 
    o.CustomerID,
    o.OrderID,
    o.ProductID,
    o.Quantity,
    o.Price,
    o.OrderDate,
    o.PromisedDeliveryDate,
    o.ActualDeliveryDate,
    o.OrderPlatform,
    o.PaymentMethod,
    o.ProductName,
    o.ProductType,
    o.Category,
    o.ItemStatus,
    o.DeliveryStatus,
    
    -- Delivery performance metrics
    DATE_DIFF(DATE(o.ActualDeliveryDate), DATE(o.PromisedDeliveryDate), DAY) AS Delivery_Delay_Days,
    
    CASE 
        WHEN DATE(o.ActualDeliveryDate) <= DATE(o.PromisedDeliveryDate) THEN 'On_Time'
        WHEN DATE(o.ActualDeliveryDate) <= DATE_ADD(DATE(o.PromisedDeliveryDate), INTERVAL 1 DAY) THEN 'Late_1_Day'
        WHEN DATE(o.ActualDeliveryDate) <= DATE_ADD(DATE(o.PromisedDeliveryDate), INTERVAL 3 DAY) THEN 'Late_2_3_Days'
        ELSE 'Late_4_Plus_Days'
    END AS Delivery_Performance,
    
    -- Order value categories
    CASE 
        WHEN (o.Quantity * o.Price) < 100 THEN 'Low_Value'
        WHEN (o.Quantity * o.Price) BETWEEN 100 AND 500 THEN 'Medium_Value'
        WHEN (o.Quantity * o.Price) BETWEEN 500 AND 1000 THEN 'High_Value'
        ELSE 'Premium_Value'
    END AS Order_Value_Category,
    
    -- Platform categorization
    CASE 
        WHEN o.OrderPlatform IN ('iOS', 'Android') THEN 'Mobile'
        WHEN o.OrderPlatform = 'Web' THEN 'Desktop'
        ELSE 'Other'
    END AS Platform_Type,
    
    -- Total order value
    (o.Quantity * o.Price) AS Total_Order_Value

FROM `customerinsightsavy.ecommerce_data.order_history` o;

-- ===============================================================
-- 3. CUSTOMER BEHAVIOR AGGREGATIONS
-- ===============================================================

-- Create customer behavior summary
CREATE OR REPLACE VIEW `customerinsightsavy.ecommerce_data.customer_behavior_summary` AS
SELECT 
    c.CustomerID,
    
    -- Order behavior
    COUNT(DISTINCT o.OrderID) as Total_Orders,
    AVG(o.Total_Order_Value) as Avg_Order_Value,
    SUM(o.Total_Order_Value) as Total_Spend_Calculated,
    MIN(DATE(o.OrderDate)) as First_Order_Date,
    MAX(DATE(o.OrderDate)) as Last_Order_Date,
    DATE_DIFF(MAX(DATE(o.OrderDate)), MIN(DATE(o.OrderDate)), DAY) as Customer_Lifespan_Days,
    
    -- Product diversity
    COUNT(DISTINCT o.Category) as Unique_Categories,
    COUNT(DISTINCT o.ProductID) as Unique_Products,
    
    -- Platform usage
    COUNTIF(o.Platform_Type = 'Mobile') as Mobile_Orders,
    COUNTIF(o.Platform_Type = 'Desktop') as Desktop_Orders,
    
    -- Delivery performance
    AVG(o.Delivery_Delay_Days) as Avg_Delivery_Delay,
    COUNTIF(o.Delivery_Performance = 'On_Time') / COUNT(*) as On_Time_Delivery_Rate,
    
    -- Payment preferences
    MODE(o.PaymentMethod) as Preferred_Payment_Method,
    
    -- Recency (days since last order)
    DATE_DIFF(CURRENT_DATE(), MAX(DATE(o.OrderDate)), DAY) as Recency_Days

FROM `customerinsightsavy.ecommerce_data.customer_features` c
LEFT JOIN `customerinsightsavy.ecommerce_data.order_analysis` o 
    ON c.CustomerID = o.CustomerID
GROUP BY c.CustomerID;

-- ===============================================================
-- 4. CUSTOMER REVIEW ANALYSIS
-- ===============================================================

-- Aggregate customer review data
CREATE OR REPLACE VIEW `customerinsightsavy.ecommerce_data.customer_review_summary` AS
SELECT 
    r.CustomerID,
    COUNT(*) as Total_Reviews,
    AVG(r.ReviewScore) as Avg_Review_Score,
    MIN(r.ReviewScore) as Min_Review_Score,
    MAX(r.ReviewScore) as Max_Review_Score,
    STDDEV(r.ReviewScore) as Review_Score_Std,
    
    -- Review sentiment categories
    COUNTIF(r.ReviewScore >= 4) as Positive_Reviews,
    COUNTIF(r.ReviewScore = 3) as Neutral_Reviews,
    COUNTIF(r.ReviewScore <= 2) as Negative_Reviews,
    
    -- Review engagement
    COUNT(r.ReviewText) / COUNT(*) as Review_Text_Rate

FROM `customerinsightsavy.ecommerce_data.customer_reviews` r
GROUP BY r.CustomerID;

-- ===============================================================
-- 5. NPS AND LOYALTY ANALYSIS
-- ===============================================================

-- NPS analysis with customer segmentation
CREATE OR REPLACE VIEW `customerinsightsavy.ecommerce_data.nps_analysis` AS
SELECT 
    n.CustomerID,
    n.SurveyCompletionDate,
    n.SurveyChannel,
    n.NPSScore,
    n.NPSType,
    n.FollowUpResponse,
    n.PurchaseFrequency,
    n.CustomerType,
    n.ProductFeedback,
    
    -- NPS categorization
    CASE 
        WHEN n.NPSScore >= 9 THEN 'Promoter'
        WHEN n.NPSScore >= 7 THEN 'Passive'
        ELSE 'Detractor'
    END AS NPS_Category,
    
    -- Loyalty scoring
    CASE 
        WHEN n.NPSScore >= 9 AND n.CustomerType = 'Returning' THEN 'High_Loyalty'
        WHEN n.NPSScore >= 7 AND n.CustomerType = 'Returning' THEN 'Medium_Loyalty'
        WHEN n.NPSScore >= 7 AND n.CustomerType = 'New' THEN 'Potential_Loyal'
        ELSE 'Low_Loyalty'
    END AS Loyalty_Segment

FROM `customerinsightsavy.ecommerce_data.nps_survey_data` n;

-- ===============================================================
-- 6. CHURN ANALYSIS
-- ===============================================================

-- Churn data with risk categorization
CREATE OR REPLACE VIEW `customerinsightsavy.ecommerce_data.churn_analysis` AS
SELECT 
    ch.CustomerID,
    ch.ReasonForLeaving,
    ch.OverallExperience,
    ch.SpecificIssues,
    ch.Suggestions,
    ch.FollowUp,
    ch.ChurnType,
    
    -- Churn risk scoring
    CASE 
        WHEN ch.ChurnType = 'Involuntary' THEN 'High_Risk'
        WHEN ch.ChurnType = 'Voluntary' AND ch.OverallExperience IN ('Very Dissatisfied', 'Dissatisfied') THEN 'High_Risk'
        WHEN ch.ChurnType = 'Engaged' AND ch.OverallExperience = 'Neutral' THEN 'Medium_Risk'
        WHEN ch.ChurnType = 'Loyal' THEN 'Low_Risk'
        ELSE 'Unknown_Risk'
    END AS Churn_Risk_Level,
    
    -- Experience scoring
    CASE 
        WHEN ch.OverallExperience = 'Very Satisfied' THEN 5
        WHEN ch.OverallExperience = 'Satisfied' THEN 4
        WHEN ch.OverallExperience = 'Neutral' THEN 3
        WHEN ch.OverallExperience = 'Dissatisfied' THEN 2
        WHEN ch.OverallExperience = 'Very Dissatisfied' THEN 1
        ELSE 0
    END AS Experience_Score

FROM `customerinsightsavy.ecommerce_data.customer_churn` ch;

-- ===============================================================
-- 7. WEBSITE ACTIVITY ANALYSIS
-- ===============================================================

-- Website engagement metrics
CREATE OR REPLACE VIEW `customerinsightsavy.ecommerce_data.website_activity_summary` AS
SELECT 
    w.CustomerID,
    AVG(w.TimeSpentOnPages) as Avg_Time_Per_Session,
    AVG(w.PagesVisited) as Avg_Pages_Per_Session,
    COUNT(*) as Total_Sessions,
    SUM(w.TimeSpentOnPages) as Total_Time_Spent,
    SUM(w.PagesVisited) as Total_Pages_Visited,
    
    -- Engagement scoring
    CASE 
        WHEN AVG(w.TimeSpentOnPages) > 10 AND AVG(w.PagesVisited) > 15 THEN 'High_Engagement'
        WHEN AVG(w.TimeSpentOnPages) > 5 AND AVG(w.PagesVisited) > 8 THEN 'Medium_Engagement'
        ELSE 'Low_Engagement'
    END AS Engagement_Level,
    
    -- Session activity types
    COUNTIF(w.SessionDetails LIKE '%purchase%' OR w.SessionDetails LIKE '%Checked out%') as Purchase_Sessions,
    COUNTIF(w.SessionDetails LIKE '%cart%') as Cart_Sessions,
    COUNTIF(w.SessionDetails LIKE '%search%') as Search_Sessions

FROM `customerinsightsavy.ecommerce_data.website_activity_logs` w
GROUP BY w.CustomerID;

-- ===============================================================
-- 8. MASTER CUSTOMER TABLE FOR ANALYTICS
-- ===============================================================

-- Create comprehensive customer analytics table
CREATE OR REPLACE TABLE `customerinsightsavy.ecommerce_data.customer_analytics_master` AS
SELECT 
    cf.*,
    
    -- Behavioral metrics
    COALESCE(cbs.Total_Orders, 0) as Orders_Count,
    COALESCE(cbs.Avg_Order_Value, 0) as Average_Order_Value,
    COALESCE(cbs.Unique_Categories, 0) as Categories_Purchased,
    COALESCE(cbs.Recency_Days, 999) as Days_Since_Last_Order,
    COALESCE(cbs.On_Time_Delivery_Rate, 0) as Delivery_Success_Rate,
    COALESCE(cbs.Mobile_Orders, 0) as Mobile_Order_Count,
    COALESCE(cbs.Desktop_Orders, 0) as Desktop_Order_Count,
    
    -- Review metrics
    COALESCE(crs.Avg_Review_Score, 3.0) as Average_Review_Score,
    COALESCE(crs.Total_Reviews, 0) as Total_Reviews_Given,
    COALESCE(crs.Positive_Reviews, 0) as Positive_Review_Count,
    
    -- NPS and loyalty
    COALESCE(nps.NPSScore, 5) as NPS_Score,
    COALESCE(nps.NPS_Category, 'Unknown') as NPS_Category,
    COALESCE(nps.Loyalty_Segment, 'Unknown') as Loyalty_Segment,
    
    -- Churn risk
    COALESCE(ca.Churn_Risk_Level, 'Unknown_Risk') as Churn_Risk,
    COALESCE(ca.Experience_Score, 3) as Customer_Experience_Score,
    COALESCE(ca.ChurnType, 'Unknown') as Churn_Status,
    
    -- Website engagement
    COALESCE(was.Avg_Time_Per_Session, 0) as Session_Time_Avg,
    COALESCE(was.Avg_Pages_Per_Session, 0) as Pages_Per_Session_Avg,
    COALESCE(was.Engagement_Level, 'Low_Engagement') as Web_Engagement_Level,
    
    -- Calculated CLV (more sophisticated)
    CASE 
        WHEN COALESCE(cbs.Total_Orders, 0) > 0 AND cf.Customer_Tenure_Days > 0
        THEN (cf.TotalSpend / cf.Customer_Tenure_Days) * 365 * 2  -- Projected 2-year CLV
        ELSE cf.TotalSpend
    END AS Calculated_CLV,
    
    -- RFM Scoring
    CASE 
        WHEN COALESCE(cbs.Recency_Days, 999) <= 30 THEN 5
        WHEN COALESCE(cbs.Recency_Days, 999) <= 60 THEN 4
        WHEN COALESCE(cbs.Recency_Days, 999) <= 90 THEN 3
        WHEN COALESCE(cbs.Recency_Days, 999) <= 180 THEN 2
        ELSE 1
    END AS Recency_Score,
    
    CASE 
        WHEN COALESCE(cbs.Total_Orders, 0) >= 20 THEN 5
        WHEN COALESCE(cbs.Total_Orders, 0) >= 15 THEN 4
        WHEN COALESCE(cbs.Total_Orders, 0) >= 10 THEN 3
        WHEN COALESCE(cbs.Total_Orders, 0) >= 5 THEN 2
        ELSE 1
    END AS Frequency_Score,
    
    CASE 
        WHEN cf.TotalSpend >= 5000 THEN 5
        WHEN cf.TotalSpend >= 2000 THEN 4
        WHEN cf.TotalSpend >= 1000 THEN 3
        WHEN cf.TotalSpend >= 500 THEN 2
        ELSE 1
    END AS Monetary_Score

FROM `customerinsightsavy.ecommerce_data.customer_features` cf
LEFT JOIN `customerinsightsavy.ecommerce_data.customer_behavior_summary` cbs 
    ON cf.CustomerID = cbs.CustomerID
LEFT JOIN `customerinsightsavy.ecommerce_data.customer_review_summary` crs 
    ON cf.CustomerID = crs.CustomerID
LEFT JOIN `customerinsightsavy.ecommerce_data.nps_analysis` nps 
    ON cf.CustomerID = nps.CustomerID
LEFT JOIN `customerinsightsavy.ecommerce_data.churn_analysis` ca 
    ON cf.CustomerID = ca.CustomerID
LEFT JOIN `customerinsightsavy.ecommerce_data.website_activity_summary` was 
    ON cf.CustomerID = was.CustomerID;

-- ===============================================================
-- 9. DATA QUALITY AND VALIDATION QUERIES
-- ===============================================================

-- Data quality check - Customer counts
SELECT 
    'Customer Data Quality Check' as Check_Type,
    COUNT(*) as Total_Records,
    COUNT(DISTINCT CustomerID) as Unique_Customers,
    COUNT(*) - COUNT(DISTINCT CustomerID) as Duplicate_Count
FROM `customerinsightsavy.ecommerce_data.customer_analytics_master`;

-- Data completeness check
SELECT 
    'Data Completeness Check' as Check_Type,
    COUNTIF(CustomerID IS NULL) as Missing_CustomerID,
    COUNTIF(Age IS NULL) as Missing_Age,
    COUNTIF(Province IS NULL) as Missing_Province,
    COUNTIF(TotalSpend IS NULL) as Missing_TotalSpend,
    COUNTIF(NPS_Score IS NULL) as Missing_NPS
FROM `customerinsightsavy.ecommerce_data.customer_analytics_master`;

-- ===============================================================
-- 10. EXPORT QUERY FOR ANALYSIS
-- ===============================================================

-- Final export query for analysis (sample)
SELECT * 
FROM `customerinsightsavy.ecommerce_data.customer_analytics_master`
WHERE CustomerID IS NOT NULL
LIMIT 1000;

-- ===============================================================
-- END OF ETL PIPELINE
-- ===============================================================
