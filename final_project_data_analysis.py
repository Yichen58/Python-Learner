"""
Final Project: Customer Analytics and Sales Prediction
Student: Data Analysis & AI Course - University of Technology Sydney

This project demonstrates:
- Core Python programming (loops, functions, lambda)
- Data manipulation with Pandas and NumPy
- Data visualization with Matplotlib and Seaborn
- Machine learning with Scikit-learn (regression and classification)
- Working with datetime and mathematical operations

Project Goal: Analyze customer data to predict sales and classify customer segments
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import datetime as dt
import math
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_customer_data(n_customers=1000):
    """
    Generate synthetic customer data for analysis
    This function demonstrates core Python concepts and data generation
    """
    print("Generating customer data...")
    
    # Create customer data using core Python and NumPy
    customers = []
    
    # Use a loop to generate customer records
    for i in range(n_customers):
        # Generate random customer attributes
        customer_id = f"CUST_{i+1:04d}"
        age = random.randint(18, 80)
        
        # Use lambda function to categorize age groups
        age_group = (lambda x: "Young" if x < 30 else "Middle" if x < 50 else "Senior")(age)
        
        # Generate income with some correlation to age
        base_income = 30000 + (age - 18) * 800 + random.normalvariate(0, 15000)
        income = max(20000, base_income)  # Minimum income floor
        
        # Generate purchase behavior
        months_as_customer = random.randint(1, 60)
        purchases_per_month = max(1, random.normalvariate(4, 2))
        avg_purchase_amount = 50 + (income / 100000) * 100 + random.normalvariate(0, 30)
        avg_purchase_amount = max(10, avg_purchase_amount)
        
        # Calculate total purchases and revenue
        total_purchases = int(months_as_customer * purchases_per_month)
        total_revenue = total_purchases * avg_purchase_amount
        
        # Generate random dates for first purchase
        start_date = dt.datetime(2020, 1, 1)
        days_offset = random.randint(0, 1460)  # Up to 4 years
        first_purchase_date = start_date + dt.timedelta(days=days_offset)
        
        # Customer segment based on revenue (will be our classification target)
        if total_revenue < 2000:
            segment = "Bronze"
        elif total_revenue < 8000:
            segment = "Silver"
        else:
            segment = "Gold"
        
        customers.append({
            'customer_id': customer_id,
            'age': age,
            'age_group': age_group,
            'income': income,
            'months_as_customer': months_as_customer,
            'total_purchases': total_purchases,
            'avg_purchase_amount': avg_purchase_amount,
            'total_revenue': total_revenue,
            'first_purchase_date': first_purchase_date,
            'segment': segment
        })
    
    return pd.DataFrame(customers)

def analyze_data(df):
    """
    Perform exploratory data analysis
    Demonstrates Pandas operations and data manipulation
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Analyze customer segments using Pandas groupby
    print(f"\nCustomer Segment Analysis:")
    segment_analysis = df.groupby('segment').agg({
        'age': 'mean',
        'income': 'mean',
        'total_revenue': ['mean', 'count'],
        'months_as_customer': 'mean'
    }).round(2)
    print(segment_analysis)
    
    # Age group analysis using lambda and Pandas
    age_group_stats = df.groupby('age_group')['total_revenue'].agg([
        'count', 
        'mean', 
        lambda x: x.quantile(0.5)  # Median using lambda
    ]).round(2)
    age_group_stats.columns = ['Count', 'Mean_Revenue', 'Median_Revenue']
    print(f"\nAge Group Analysis:")
    print(age_group_stats)
    
    return df

def create_visualizations(df):
    """
    Create data visualizations using Matplotlib and Seaborn
    """
    print("\nCreating visualizations...")
    
    # Set style for better-looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Customer Analytics Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Age distribution histogram
    axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Income vs Revenue scatter plot
    axes[0, 1].scatter(df['income'], df['total_revenue'], alpha=0.6, c='coral')
    axes[0, 1].set_title('Income vs Total Revenue')
    axes[0, 1].set_xlabel('Income ($)')
    axes[0, 1].set_ylabel('Total Revenue ($)')
    
    # 3. Customer segment pie chart
    segment_counts = df['segment'].value_counts()
    axes[0, 2].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
    axes[0, 2].set_title('Customer Segment Distribution')
    
    # 4. Revenue by age group box plot
    sns.boxplot(data=df, x='age_group', y='total_revenue', ax=axes[1, 0])
    axes[1, 0].set_title('Revenue Distribution by Age Group')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Monthly trends - extract month from first purchase date
    df['first_purchase_month'] = df['first_purchase_date'].dt.month
    monthly_revenue = df.groupby('first_purchase_month')['total_revenue'].mean()
    axes[1, 1].plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2)
    axes[1, 1].set_title('Average Revenue by First Purchase Month')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Average Revenue ($)')
    
    # 6. Correlation heatmap
    numeric_cols = ['age', 'income', 'months_as_customer', 'total_purchases', 'total_revenue']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
    axes[1, 2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('customer_analytics_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'customer_analytics_dashboard.png'")

def revenue_prediction_model(df):
    """
    Build a regression model to predict customer revenue
    Demonstrates Scikit-learn regression
    """
    print("\n" + "="*50)
    print("REVENUE PREDICTION MODEL (REGRESSION)")
    print("="*50)
    
    # Prepare features for regression
    feature_columns = ['age', 'income', 'months_as_customer', 'total_purchases']
    X = df[feature_columns]
    y = df['total_revenue']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    
    print(f"Model Performance:")
    print(f"Root Mean Square Error: ${rmse:.2f}")
    print(f"R-squared Score: {model.score(X_test, y_test):.3f}")
    
    # Display feature importance (coefficients)
    print(f"\nFeature Coefficients:")
    for feature, coef in zip(feature_columns, model.coef_):
        print(f"{feature}: {coef:.2f}")
    
    # Make predictions for new customers
    print(f"\nSample Predictions for New Customers:")
    sample_customers = [
        [25, 45000, 12, 48],  # Young customer
        [40, 75000, 24, 96],  # Middle-aged customer
        [60, 65000, 36, 144]  # Senior customer
    ]
    
    for i, customer in enumerate(sample_customers, 1):
        predicted_revenue = model.predict([customer])[0]
        print(f"Customer {i} (Age: {customer[0]}, Income: ${customer[1]}, "
              f"Months: {customer[2]}, Purchases: {customer[3]}) "
              f"-> Predicted Revenue: ${predicted_revenue:.2f}")
    
    return model

def customer_segmentation_model(df):
    """
    Build a classification model to predict customer segments
    Demonstrates Scikit-learn classification
    """
    print("\n" + "="*50)
    print("CUSTOMER SEGMENTATION MODEL (CLASSIFICATION)")
    print("="*50)
    
    # Prepare features for classification
    feature_columns = ['age', 'income', 'months_as_customer', 'total_purchases', 'avg_purchase_amount']
    X = df[feature_columns]
    y = df['segment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.3f}")
    
    # Display detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print(f"\nFeature Importance:")
    for feature, importance in zip(feature_columns, model.feature_importances_):
        print(f"{feature}: {importance:.3f}")
    
    # Predict segments for sample customers
    print(f"\nSample Segment Predictions:")
    sample_customers = [
        [25, 35000, 6, 24, 45],   # Likely Bronze
        [35, 65000, 18, 72, 85],  # Likely Silver
        [45, 95000, 30, 150, 120] # Likely Gold
    ]
    
    for i, customer in enumerate(sample_customers, 1):
        predicted_segment = model.predict([customer])[0]
        probability = model.predict_proba([customer])[0]
        max_prob = max(probability)
        print(f"Customer {i} -> Predicted Segment: {predicted_segment} "
              f"(Confidence: {max_prob:.2f})")
    
    return model

def datetime_analysis(df):
    """
    Demonstrate datetime operations and time-based analysis
    """
    print("\n" + "="*50)
    print("DATETIME ANALYSIS")
    print("="*50)
    
    # Current date for calculations
    current_date = dt.datetime.now()
    
    # Calculate customer lifetime in days
    df['days_since_first_purchase'] = (current_date - df['first_purchase_date']).dt.days
    
    # Extract various date components
    df['first_purchase_year'] = df['first_purchase_date'].dt.year
    df['first_purchase_quarter'] = df['first_purchase_date'].dt.quarter
    df['day_of_week'] = df['first_purchase_date'].dt.day_name()
    
    # Analyze seasonal patterns
    print("Customer Acquisition by Quarter:")
    quarterly_customers = df.groupby(['first_purchase_year', 'first_purchase_quarter']).size()
    print(quarterly_customers.tail(8))
    
    # Day of week analysis
    print(f"\nCustomer Acquisition by Day of Week:")
    dow_analysis = df['day_of_week'].value_counts()
    print(dow_analysis)
    
    # Calculate customer lifetime value rate
    df['revenue_per_day'] = df['total_revenue'] / df['days_since_first_purchase']
    
    print(f"\nTop 5 Customers by Daily Revenue Rate:")
    top_daily_revenue = df.nlargest(5, 'revenue_per_day')[['customer_id', 'revenue_per_day', 'total_revenue']]
    print(top_daily_revenue)

def mathematical_operations(df):
    """
    Demonstrate mathematical operations and calculations
    """
    print("\n" + "="*50)
    print("MATHEMATICAL ANALYSIS")
    print("="*50)
    
    # Calculate various mathematical metrics
    # Use mathematical functions for advanced calculations
    
    # Customer value score using logarithmic scaling
    df['value_score'] = df['total_revenue'].apply(lambda x: math.log(x + 1))
    
    # Purchase frequency score using square root
    df['frequency_score'] = df['total_purchases'].apply(lambda x: math.sqrt(x))
    
    # Recency score (inverse of days since first purchase)
    max_days = df['days_since_first_purchase'].max()
    df['recency_score'] = df['days_since_first_purchase'].apply(
        lambda x: math.exp(-(x / max_days))
    )
    
    # Combined RFM-like score
    df['combined_score'] = (df['value_score'] * 0.4 + 
                           df['frequency_score'] * 0.3 + 
                           df['recency_score'] * 0.3)
    
    print("Top 10 Customers by Combined Score:")
    top_customers = df.nlargest(10, 'combined_score')[
        ['customer_id', 'segment', 'combined_score', 'total_revenue']
    ]
    print(top_customers)
    
    # Statistical calculations
    print(f"\nStatistical Summary:")
    print(f"Average Revenue: ${df['total_revenue'].mean():.2f}")
    print(f"Revenue Standard Deviation: ${df['total_revenue'].std():.2f}")
    print(f"Revenue Median: ${df['total_revenue'].median():.2f}")
    print(f"Revenue 95th Percentile: ${df['total_revenue'].quantile(0.95):.2f}")

def main():
    """
    Main function to orchestrate the entire analysis
    Demonstrates function organization and program structure
    """
    print("="*60)
    print("CUSTOMER ANALYTICS AND SALES PREDICTION PROJECT")
    print("University of Technology Sydney - Data Analysis & AI")
    print("="*60)
    
    # Step 1: Generate synthetic data
    df = generate_customer_data(n_customers=1000)
    
    # Step 2: Exploratory Data Analysis
    df = analyze_data(df)
    
    # Step 3: Create visualizations
    create_visualizations(df)
    
    # Step 4: Build regression model for revenue prediction
    revenue_model = revenue_prediction_model(df)
    
    # Step 5: Build classification model for customer segmentation
    segmentation_model = customer_segmentation_model(df)
    
    # Step 6: Perform datetime analysis
    datetime_analysis(df)
    
    # Step 7: Mathematical operations and scoring
    mathematical_operations(df)
    
    # Final summary
    print("\n" + "="*60)
    print("PROJECT SUMMARY")
    print("="*60)
    print("Successfully completed comprehensive data analysis including:")
    print("✓ Data generation and manipulation with Pandas/NumPy")
    print("✓ Exploratory data analysis with statistical summaries")
    print("✓ Data visualization with Matplotlib and Seaborn")
    print("✓ Machine learning models (regression and classification)")
    print("✓ Datetime operations and time-based analysis")
    print("✓ Mathematical operations and scoring algorithms")
    print("✓ Core Python programming concepts (loops, functions, lambda)")
    
    # Return the final dataset for further analysis if needed
    return df

# Execute the main function when script is run
if __name__ == "__main__":
    # This demonstrates proper Python script structure
    final_dataset = main()
    print(f"\nFinal dataset shape: {final_dataset.shape}")
    print("Analysis complete! Check 'customer_analytics_dashboard.png' for visualizations.")