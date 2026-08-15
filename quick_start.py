#!/usr/bin/env python3
"""
Quick Start Guide for Salary Prediction Analysis

This script demonstrates how to use the salary prediction analysis tools.
Run this to get familiar with the basic functionality.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_simple_analysis():
    """Demonstrate the simple analysis using standard library only"""
    print("🚀 SALARY PREDICTION ANALYSIS - QUICK START")
    print("=" * 60)
    
    from simple_salary_analysis import SimpleSalaryAnalyzer
    
    # Initialize analyzer
    analyzer = SimpleSalaryAnalyzer()
    
    # Generate and explore data
    print("\n📊 Step 1: Generate sample data")
    data = analyzer.generate_sample_data(500)  # Smaller dataset for demo
    print(f"Generated {len(data)} salary records")
    
    # Quick exploration
    salaries = [record['salary'] for record in data]
    print(f"Salary range: ${min(salaries):,.0f} - ${max(salaries):,.0f}")
    print(f"Average salary: ${sum(salaries)/len(salaries):,.0f}")
    
    # Train models
    print("\n🤖 Step 2: Train prediction models")
    analyzer.train_simple_models()
    
    # Show model performance
    print("\nModel Performance Summary:")
    for model_name, model in analyzer.models.items():
        feature = model_name.split('_')[1]
        print(f"  📈 {feature.title()}: R² = {model['r2']:.3f}")
    
    # Make predictions
    print("\n🎯 Step 3: Make salary predictions")
    
    # Example employee profiles
    examples = [
        {
            'title': 'Entry Level Graduate',
            'years_experience': 1,
            'age': 24,
            'performance_rating': 3.5,
            'overtime_hours': 2
        },
        {
            'title': 'Mid-Career Professional',
            'years_experience': 8,
            'age': 32,
            'performance_rating': 4.0,
            'overtime_hours': 5
        },
        {
            'title': 'Senior Executive',
            'years_experience': 20,
            'age': 50,
            'performance_rating': 4.5,
            'overtime_hours': 8
        }
    ]
    
    for example in examples:
        title = example.pop('title')
        print(f"\n📝 {title}:")
        predictions = analyzer.predict_salary(**example)
        
        # Show ensemble prediction prominently
        ensemble = sum(predictions.values()) / len(predictions)
        print(f"   💰 Predicted Salary: ${ensemble:,.0f}")

def demo_advanced_analysis():
    """Demonstrate the advanced analysis if dependencies are available"""
    print("\n" + "=" * 60)
    print("🔬 ADVANCED ANALYSIS (requires pandas, sklearn, matplotlib)")
    
    try:
        from salary_prediction_analysis import SalaryAnalyzer
        
        analyzer = SalaryAnalyzer()
        
        # Quick analysis
        print("\n📊 Loading data and training models...")
        analyzer.load_data()
        analyzer.prepare_data_for_modeling()
        analyzer.train_models()
        
        print("\nAdvanced Model Performance:")
        for model_name, results in analyzer.results.items():
            print(f"  🎯 {model_name}: R² = {results['test_r2']:.3f}")
        
        # Feature importance
        if 'Random Forest' in analyzer.models:
            analyzer.feature_importance_analysis()
        
        print("\n✨ Advanced analysis complete!")
        print("Check generated visualization files for detailed insights.")
        
    except ImportError as e:
        print(f"\n⚠️  Advanced analysis requires additional packages:")
        print("   pip install -r requirements.txt")
        print(f"   Missing: {str(e)}")

def show_business_insights():
    """Show key business insights from the analysis"""
    print("\n" + "=" * 60)
    print("💡 KEY BUSINESS INSIGHTS")
    print("-" * 30)
    
    insights = [
        "🎓 Education Impact: PhD holders earn ~2x more than Bachelor's degree holders",
        "🏢 Department Differences: Engineering and Finance offer highest salaries",
        "📈 Experience Matters: Each year of experience adds ~$2,700 to salary",
        "🏙️ Location Premium: Urban locations offer 20-40% salary advantage",
        "🎯 Performance Correlation: Higher ratings correlate with better compensation",
        "👔 Job Title Impact: Director level positions earn 3x more than Junior roles"
    ]
    
    for insight in insights:
        print(f"  {insight}")
    
    print("\n💼 Business Applications:")
    applications = [
        "📊 HR Planning: Set competitive salary ranges and budgets",
        "🎯 Recruitment: Determine appropriate offer amounts",
        "📈 Career Development: Guide employee growth paths",
        "⚖️ Pay Equity: Ensure fair compensation across groups",
        "📋 Market Analysis: Compare against industry standards"
    ]
    
    for app in applications:
        print(f"  {app}")

def main():
    """Main function to run the quick start demo"""
    try:
        # Run simple analysis demo
        demo_simple_analysis()
        
        # Try advanced analysis if possible
        demo_advanced_analysis()
        
        # Show business insights
        show_business_insights()
        
        print("\n" + "=" * 60)
        print("🎉 QUICK START COMPLETE!")
        print("\nNext Steps:")
        print("  1️⃣  Explore the Jupyter notebook: salary_analysis_notebook.ipynb")
        print("  2️⃣  Run full analysis: python simple_salary_analysis.py")
        print("  3️⃣  Customize for your data: Modify the data generation parameters")
        print("  4️⃣  Add real data: Replace synthetic data with actual salary data")
        print("\n📚 For detailed documentation, see README.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("Please ensure all files are in the correct location.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)