🚀 Customer Retention & Churn Analysis

    Analyze e-commerce customer data to identify loyal vs. churned customers using RFM metrics and predict churn using a Random Forest Classifier. Gain actionable insights and visualize key patterns.

📂 Project Files

    main.py – Python code for data cleaning, RFM calculation, churn analysis, and ML model

    OnlineRetail_sample.csv – Sample dataset for analysis

    customer_churn_analysis.csv – Processed dataset with RFM metrics and churn label

⚡ How to Run

   Install libraries:

   pip install pandas matplotlib seaborn scikit-learn


   Place OnlineRetail_sample.csv in the same folder as main.py.

   Run the code:

   python main.py


   Outputs:

📊 Histograms of Recency & Monetary

🌟 Feature Importance chart from Random Forest

💾 customer_churn_analysis.csv

🛠 Project Steps

Data Cleaning – Remove missing CustomerIDs, negative quantities, calculate TotalPrice

RFM Metrics – Recency, Frequency, Monetary

Churn Labeling – Customers with Recency > 90 days → Churn = 1

Exploratory Analysis – Visualize distributions, calculate churn rate

Machine Learning – Random Forest to predict churn

Feature Importance – Identify which metric impacts churn most

Save Processed Data – Ready for dashboards or further analysis

📈 Insights

Customers with high Recency & low Frequency are most likely to churn

Monetary value is less predictive than Recency

Churn rate helps plan retention strategies and marketing campaigns

🎨 Visuals

Recency & Monetary Distribution:


Feature Importance:


💡 Notes

Sample dataset used due to size limits; full dataset can be processed locally

Fully Python-based; no Tableau/Power BI needed

Modular code – easy to extend for advanced ML or analytics
