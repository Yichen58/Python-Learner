"""
Test script for the Salary Prediction Analysis
This script validates the functionality of the salary analyzer
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from salary_prediction_analysis import SalaryAnalyzer

def test_basic_functionality():
    """Test basic functionality of the SalaryAnalyzer"""
    print("Testing Salary Prediction Analysis...")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = SalaryAnalyzer()
        print("✓ SalaryAnalyzer initialized successfully")
        
        # Test data generation
        data = analyzer.generate_sample_data(100)  # Small dataset for testing
        assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"
        assert len(data) == 100, "Should generate 100 samples"
        assert 'salary' in data.columns, "Should have salary column"
        print("✓ Sample data generation works")
        
        # Test data loading
        analyzer.load_data()
        assert analyzer.data is not None, "Data should be loaded"
        print("✓ Data loading works")
        
        # Test data preparation
        X_train, X_test, y_train, y_test = analyzer.prepare_data_for_modeling()
        assert X_train is not None, "Training features should exist"
        assert len(X_train) > 0, "Training set should not be empty"
        print("✓ Data preparation works")
        
        # Test model training
        analyzer.train_models()
        assert len(analyzer.models) > 0, "Should have trained models"
        assert 'Linear Regression' in analyzer.models, "Should have Linear Regression model"
        print("✓ Model training works")
        
        # Test predictions
        predictions = analyzer.predict_salary(
            age=30,
            years_experience=5,
            department='Engineering',
            education_level='Bachelor',
            job_title='Mid-level',
            performance_rating=3.8,
            overtime_hours=3,
            location='Urban'
        )
        assert isinstance(predictions, dict), "Predictions should be a dictionary"
        assert len(predictions) > 0, "Should have at least one prediction"
        print("✓ Salary prediction works")
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        return False

def test_data_quality():
    """Test the quality of generated data"""
    print("\nTesting data quality...")
    
    analyzer = SalaryAnalyzer()
    data = analyzer.generate_sample_data(1000)
    
    # Check data types
    assert data['age'].dtype in [np.int64, int], "Age should be integer"
    assert data['salary'].dtype in [np.float64, float], "Salary should be numeric"
    
    # Check data ranges
    assert data['age'].min() >= 22, "Minimum age should be 22"
    assert data['age'].max() <= 65, "Maximum age should be 65"
    assert data['salary'].min() > 0, "Salary should be positive"
    
    # Check for missing values
    assert data.isnull().sum().sum() == 0, "Should have no missing values"
    
    print("✓ Data quality checks passed")

def quick_demo():
    """Run a quick demonstration of the analysis"""
    print("\nRunning quick demonstration...")
    print("-" * 30)
    
    analyzer = SalaryAnalyzer()
    
    # Load data
    data = analyzer.load_data()
    print(f"Loaded {len(data)} samples")
    
    # Show basic statistics
    print(f"Salary range: ${data['salary'].min():,.0f} - ${data['salary'].max():,.0f}")
    print(f"Average salary: ${data['salary'].mean():,.0f}")
    
    # Prepare and train models
    analyzer.prepare_data_for_modeling()
    analyzer.train_models()
    
    # Show model performance
    print("\nModel Performance:")
    for model_name, results in analyzer.results.items():
        print(f"  {model_name}: R² = {results['test_r2']:.3f}")
    
    # Make a sample prediction
    print("\nSample Prediction (Senior Engineer):")
    predictions = analyzer.predict_salary(
        age=35,
        years_experience=8,
        department='Engineering',
        education_level='Master',
        job_title='Senior',
        performance_rating=4.2,
        overtime_hours=5,
        location='Urban'
    )

if __name__ == "__main__":
    # Run tests
    success = test_basic_functionality()
    
    if success:
        test_data_quality()
        quick_demo()
        print("\n🎉 All tests completed successfully!")
        print("The Salary Prediction Analysis is ready to use.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)