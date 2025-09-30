"""
Test script for the Simple Salary Prediction Analysis
This script validates the basic functionality using only standard library
"""

import sys
import os

# Add current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_salary_analysis import SimpleSalaryAnalyzer

def test_basic_functionality():
    """Test basic functionality of the SimpleSalaryAnalyzer"""
    print("Testing Simple Salary Prediction Analysis...")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = SimpleSalaryAnalyzer()
        print("✓ SimpleSalaryAnalyzer initialized successfully")
        
        # Test data generation
        data = analyzer.generate_sample_data(100)  # Small dataset for testing
        assert isinstance(data, list), "Data should be a list"
        assert len(data) == 100, "Should generate 100 samples"
        assert 'salary' in data[0], "Should have salary field"
        print("✓ Sample data generation works")
        
        # Test data loading
        analyzer.load_data()
        assert analyzer.data is not None, "Data should be loaded"
        print("✓ Data loading works")
        
        # Test data exploration
        analyzer.explore_data()
        print("✓ Data exploration works")
        
        # Test statistics calculation
        analyzer.calculate_statistics()
        print("✓ Statistics calculation works")
        
        # Test model training
        analyzer.train_simple_models()
        assert len(analyzer.models) > 0, "Should have trained models"
        print("✓ Model training works")
        
        # Test predictions
        predictions = analyzer.predict_salary(
            years_experience=5,
            age=30,
            performance_rating=3.8,
            overtime_hours=3
        )
        assert isinstance(predictions, dict), "Predictions should be a dictionary"
        assert len(predictions) > 0, "Should have at least one prediction"
        print("✓ Salary prediction works")
        
        print("\n" + "=" * 50)
        print("All basic tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_quality():
    """Test the quality of generated data"""
    print("\nTesting data quality...")
    
    analyzer = SimpleSalaryAnalyzer()
    data = analyzer.generate_sample_data(1000)
    
    # Check data structure
    assert isinstance(data, list), "Data should be a list"
    assert len(data) > 0, "Should have data"
    assert isinstance(data[0], dict), "Each record should be a dictionary"
    
    # Check required fields
    required_fields = ['age', 'years_experience', 'salary', 'department', 'education_level']
    for field in required_fields:
        assert field in data[0], f"Should have {field} field"
    
    # Check data types and ranges
    for record in data[:10]:  # Check first 10 records
        assert isinstance(record['age'], int), "Age should be integer"
        assert isinstance(record['salary'], (int, float)), "Salary should be numeric"
        assert 22 <= record['age'] <= 65, "Age should be in valid range"
        assert record['salary'] > 0, "Salary should be positive"
    
    print("✓ Data quality checks passed")

def quick_demo():
    """Run a quick demonstration of the analysis"""
    print("\nRunning quick demonstration...")
    print("-" * 30)
    
    analyzer = SimpleSalaryAnalyzer()
    
    # Load data
    data = analyzer.load_data()
    print(f"Loaded {len(data)} samples")
    
    # Show basic statistics
    salaries = [record['salary'] for record in data]
    print(f"Salary range: ${min(salaries):,.0f} - ${max(salaries):,.0f}")
    print(f"Average salary: ${sum(salaries)/len(salaries):,.0f}")
    
    # Train models
    analyzer.train_simple_models()
    
    # Show model performance
    print("\nModel Performance:")
    for model_name, model in analyzer.models.items():
        feature = model_name.split('_')[1]
        print(f"  {feature}: R² = {model['r2']:.3f}")
    
    # Make a sample prediction
    print("\nSample Prediction (Mid-level Professional):")
    analyzer.predict_salary(
        years_experience=5,
        age=30,
        performance_rating=3.8,
        overtime_hours=4
    )

if __name__ == "__main__":
    # Run tests
    success = test_basic_functionality()
    
    if success:
        test_data_quality()
        quick_demo()
        print("\n🎉 All tests completed successfully!")
        print("The Simple Salary Prediction Analysis is ready to use.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)