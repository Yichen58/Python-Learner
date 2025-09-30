"""
Simplified Salary Prediction Analysis
A basic implementation using only Python standard library for demonstration

This module demonstrates Python data analysis concepts without external dependencies
"""

import random
import json
import csv
from collections import defaultdict, Counter
import math
import os

class SimpleSalaryAnalyzer:
    """
    A simplified class to perform basic salary prediction analysis
    using only Python standard library
    """
    
    def __init__(self):
        """Initialize the SimpleSalaryAnalyzer"""
        self.data = []
        self.models = {}
        self.results = {}
        
    def generate_sample_data(self, n_samples=1000):
        """
        Generate synthetic salary data for analysis
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            list: Generated salary dataset
        """
        random.seed(42)
        
        # Define possible values for categorical variables
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
        education_levels = ['Bachelor', 'Master', 'PhD', 'High School']
        job_titles = ['Junior', 'Mid-level', 'Senior', 'Lead', 'Manager', 'Director']
        locations = ['Urban', 'Suburban', 'Rural']
        
        data = []
        
        for i in range(n_samples):
            age = random.randint(22, 65)
            years_experience = random.randint(0, 40)
            department = random.choice(departments)
            education_level = random.choice(education_levels)
            job_title = random.choice(job_titles)
            performance_rating = random.uniform(1, 5)
            overtime_hours = random.randint(0, 20)
            location = random.choice(locations)
            
            # Generate salary based on realistic relationships
            salary_base = 40000
            
            # Age factor (peak earning in middle age)
            age_factor = age * 500 if age < 40 else (65 - age) * 300
            
            # Experience factor
            exp_factor = years_experience * 1200
            
            # Department factor
            dept_multiplier = {
                'Engineering': 1.3, 'Sales': 1.1, 'Marketing': 1.0,
                'HR': 0.9, 'Finance': 1.2, 'Operations': 0.95
            }
            dept_factor = dept_multiplier[department]
            
            # Education factor
            edu_multiplier = {
                'High School': 0.8, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.6
            }
            edu_factor = edu_multiplier[education_level]
            
            # Job title factor
            title_multiplier = {
                'Junior': 0.7, 'Mid-level': 1.0, 'Senior': 1.4,
                'Lead': 1.6, 'Manager': 1.8, 'Director': 2.2
            }
            title_factor = title_multiplier[job_title]
            
            # Performance factor
            perf_factor = performance_rating * 3000
            
            # Location factor
            loc_multiplier = {'Urban': 1.2, 'Suburban': 1.0, 'Rural': 0.85}
            loc_factor = loc_multiplier[location]
            
            # Calculate salary with some random noise
            salary = (
                salary_base + 
                age_factor + 
                exp_factor + 
                perf_factor + 
                (overtime_hours * 500) +
                random.gauss(0, 5000)
            )
            
            # Apply multipliers
            salary = salary * dept_factor * edu_factor * title_factor * loc_factor
            
            # Ensure positive salaries
            salary = max(salary, 25000)
            
            # Round salaries
            salary = round(salary, 0)
            
            record = {
                'age': age,
                'years_experience': years_experience,
                'department': department,
                'education_level': education_level,
                'job_title': job_title,
                'performance_rating': round(performance_rating, 2),
                'overtime_hours': overtime_hours,
                'location': location,
                'salary': salary
            }
            
            data.append(record)
        
        self.data = data
        return self.data
    
    def load_data(self, filepath=None):
        """
        Load salary data from file or generate sample data
        
        Args:
            filepath (str): Path to CSV file, if None generates sample data
        """
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
                # Convert numeric fields
                for record in self.data:
                    record['age'] = int(record['age'])
                    record['years_experience'] = int(record['years_experience'])
                    record['performance_rating'] = float(record['performance_rating'])
                    record['overtime_hours'] = int(record['overtime_hours'])
                    record['salary'] = float(record['salary'])
        else:
            self.data = self.generate_sample_data()
        
        print(f"Data loaded successfully! Shape: {len(self.data)} samples, {len(self.data[0]) if self.data else 0} features")
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        if not self.data:
            raise ValueError("No data loaded. Please load data first.")
        
        print("=== SALARY PREDICTION ANALYSIS ===\n")
        print("1. DATASET OVERVIEW")
        print(f"Dataset size: {len(self.data)} samples")
        print(f"Features: {list(self.data[0].keys()) if self.data else 'None'}")
        
        # Show first 5 rows
        print("\nFirst 5 rows:")
        for i, record in enumerate(self.data[:5]):
            print(f"Row {i+1}: {record}")
        
        # Calculate basic statistics
        print("\n2. SALARY STATISTICS")
        salaries = [record['salary'] for record in self.data]
        print(f"Count: {len(salaries)}")
        print(f"Mean: ${sum(salaries)/len(salaries):,.2f}")
        print(f"Min: ${min(salaries):,.2f}")
        print(f"Max: ${max(salaries):,.2f}")
        print(f"Median: ${sorted(salaries)[len(salaries)//2]:,.2f}")
        
        # Categorical variable distributions
        print("\n3. CATEGORICAL VARIABLES DISTRIBUTION")
        categorical_cols = ['department', 'education_level', 'job_title', 'location']
        
        for col in categorical_cols:
            values = [record[col] for record in self.data]
            counter = Counter(values)
            print(f"\n{col}:")
            for value, count in counter.most_common():
                print(f"  {value}: {count} ({count/len(values)*100:.1f}%)")
    
    def calculate_statistics(self):
        """Calculate correlation and other statistics"""
        if not self.data:
            raise ValueError("No data loaded. Please load data first.")
        
        print("\n4. CORRELATION ANALYSIS")
        
        # Calculate salary by category
        categories = ['department', 'education_level', 'job_title', 'location']
        
        for category in categories:
            category_salary = defaultdict(list)
            for record in self.data:
                category_salary[record[category]].append(record['salary'])
            
            print(f"\nAverage salary by {category}:")
            for cat_value, salaries in category_salary.items():
                avg_salary = sum(salaries) / len(salaries)
                print(f"  {cat_value}: ${avg_salary:,.2f}")
        
        # Numeric correlations (simplified)
        print("\nNumeric variable relationships:")
        numeric_vars = ['age', 'years_experience', 'performance_rating', 'overtime_hours']
        
        for var in numeric_vars:
            # Simple correlation calculation
            x_values = [record[var] for record in self.data]
            y_values = [record['salary'] for record in self.data]
            
            # Calculate Pearson correlation coefficient
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            sum_y2 = sum(y * y for y in y_values)
            
            correlation = (n * sum_xy - sum_x * sum_y) / math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
            print(f"  {var} vs salary: {correlation:.3f}")
    
    def simple_linear_regression(self, x_values, y_values):
        """
        Simple linear regression implementation
        
        Args:
            x_values: Feature values
            y_values: Target values
            
        Returns:
            tuple: (slope, intercept)
        """
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    def predict_with_linear_model(self, x_value, slope, intercept):
        """Predict using linear model"""
        return slope * x_value + intercept
    
    def calculate_r_squared(self, actual, predicted):
        """Calculate R-squared value"""
        mean_actual = sum(actual) / len(actual)
        ss_tot = sum((y - mean_actual)**2 for y in actual)
        ss_res = sum((y - p)**2 for y, p in zip(actual, predicted))
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def train_simple_models(self):
        """
        Train simple models for salary prediction
        """
        if not self.data:
            self.load_data()
        
        print("\n=== MODEL TRAINING ===")
        
        # Prepare data for modeling
        # Split data into training (80%) and testing (20%)
        random.shuffle(self.data)
        split_idx = int(len(self.data) * 0.8)
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        
        print(f"Training set size: {len(train_data)}")
        print(f"Test set size: {len(test_data)}")
        
        # Train models for numeric features
        numeric_features = ['years_experience', 'age', 'performance_rating', 'overtime_hours']
        
        for feature in numeric_features:
            # Extract training data
            x_train = [record[feature] for record in train_data]
            y_train = [record['salary'] for record in train_data]
            
            # Train linear regression
            slope, intercept = self.simple_linear_regression(x_train, y_train)
            
            # Make predictions on test set
            x_test = [record[feature] for record in test_data]
            y_test = [record['salary'] for record in test_data]
            y_pred = [self.predict_with_linear_model(x, slope, intercept) for x in x_test]
            
            # Calculate R-squared
            r2 = self.calculate_r_squared(y_test, y_pred)
            
            # Calculate MSE
            mse = sum((y - p)**2 for y, p in zip(y_test, y_pred)) / len(y_test)
            
            # Store model
            self.models[f'Linear_{feature}'] = {
                'slope': slope,
                'intercept': intercept,
                'r2': r2,
                'mse': mse
            }
            
            print(f"\n{feature} model:")
            print(f"  R²: {r2:.4f}")
            print(f"  MSE: {mse:,.2f}")
            print(f"  Equation: salary = {slope:.2f} * {feature} + {intercept:.2f}")
    
    def predict_salary(self, years_experience=5, age=30, performance_rating=3.5, overtime_hours=5):
        """
        Predict salary using trained models
        
        Args:
            years_experience (int): Years of experience
            age (int): Employee age
            performance_rating (float): Performance rating (1-5)
            overtime_hours (int): Overtime hours per month
            
        Returns:
            dict: Predictions from different models
        """
        if not self.models:
            self.train_simple_models()
        
        predictions = {}
        
        # Make predictions using each model
        for model_name, model in self.models.items():
            feature = model_name.split('_')[1]  # Extract feature name
            pred = 25000  # Default minimum salary
            
            if feature == 'years_experience':
                pred = self.predict_with_linear_model(years_experience, model['slope'], model['intercept'])
            elif feature == 'age':
                pred = self.predict_with_linear_model(age, model['slope'], model['intercept'])
            elif feature == 'performance_rating':
                pred = self.predict_with_linear_model(performance_rating, model['slope'], model['intercept'])
            elif feature == 'overtime_hours':
                pred = self.predict_with_linear_model(overtime_hours, model['slope'], model['intercept'])
            
            predictions[model_name] = max(pred, 25000)  # Minimum salary
        
        print(f"\n=== SALARY PREDICTION ===")
        print(f"Employee Profile:")
        print(f"  Years Experience: {years_experience}")
        print(f"  Age: {age}")
        print(f"  Performance Rating: {performance_rating}/5")
        print(f"  Overtime Hours: {overtime_hours}")
        print(f"\nPredicted Salaries:")
        
        for model_name, pred_salary in predictions.items():
            feature = model_name.split('_')[1]
            r2 = self.models[model_name]['r2']
            print(f"  Based on {feature}: ${pred_salary:,.2f} (R² = {r2:.3f})")
        
        # Simple ensemble prediction (average of all models)
        ensemble_pred = sum(predictions.values()) / len(predictions)
        print(f"  Ensemble Average: ${ensemble_pred:,.2f}")
        
        return predictions
    
    def save_data_to_csv(self, filepath='salary_data.csv'):
        """Save generated data to CSV file"""
        if not self.data:
            raise ValueError("No data to save. Please load or generate data first.")
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)
        
        print(f"Data saved to {filepath}")
    
    def save_results_to_json(self, filepath='analysis_results.json'):
        """Save analysis results to JSON file"""
        results = {
            'data_summary': {
                'sample_size': len(self.data),
                'features': list(self.data[0].keys()) if self.data else []
            },
            'models': self.models
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def run_complete_analysis(self):
        """
        Run the complete salary prediction analysis pipeline
        """
        print("Starting Basic Salary Prediction Analysis...")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Calculate statistics
        self.calculate_statistics()
        
        # Step 4: Train models
        self.train_simple_models()
        
        # Step 5: Make example predictions
        print("\n=== EXAMPLE PREDICTIONS ===")
        
        # Different employee profiles
        test_cases = [
            {'years_experience': 2, 'age': 25, 'performance_rating': 3.5, 'overtime_hours': 2},
            {'years_experience': 8, 'age': 35, 'performance_rating': 4.2, 'overtime_hours': 5},
            {'years_experience': 15, 'age': 45, 'performance_rating': 4.8, 'overtime_hours': 8}
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            self.predict_salary(**case)
        
        # Step 6: Save results
        self.save_data_to_csv()
        self.save_results_to_json()
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("Generated files:")
        print("- salary_data.csv (dataset)")
        print("- analysis_results.json (model results)")


def main():
    """
    Main function to run the salary prediction analysis
    """
    analyzer = SimpleSalaryAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()