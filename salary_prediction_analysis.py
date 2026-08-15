"""
Salary Prediction Analysis
A comprehensive analysis and prediction model for salary data

This module demonstrates Python data science techniques including:
- Data generation and manipulation
- Exploratory data analysis
- Data visualization
- Machine learning model building and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SalaryAnalyzer:
    """
    A class to perform comprehensive salary prediction analysis
    """
    
    def __init__(self):
        """Initialize the SalaryAnalyzer"""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def generate_sample_data(self, n_samples=1000):
        """
        Generate synthetic salary data for analysis
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pandas.DataFrame: Generated salary dataset
        """
        np.random.seed(42)
        
        # Define possible values for categorical variables
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
        education_levels = ['Bachelor', 'Master', 'PhD', 'High School']
        job_titles = ['Junior', 'Mid-level', 'Senior', 'Lead', 'Manager', 'Director']
        
        data = {
            'age': np.random.randint(22, 65, n_samples),
            'years_experience': np.random.randint(0, 40, n_samples),
            'department': np.random.choice(departments, n_samples),
            'education_level': np.random.choice(education_levels, n_samples),
            'job_title': np.random.choice(job_titles, n_samples),
            'performance_rating': np.random.uniform(1, 5, n_samples),
            'overtime_hours': np.random.randint(0, 20, n_samples),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
        }
        
        # Generate salary based on realistic relationships
        salary_base = 40000
        
        # Age factor (peak earning in middle age)
        age_factor = np.where(data['age'] < 40, data['age'] * 500, (65 - data['age']) * 300)
        
        # Experience factor
        exp_factor = data['years_experience'] * 1200
        
        # Department factor
        dept_multiplier = {
            'Engineering': 1.3, 'Sales': 1.1, 'Marketing': 1.0,
            'HR': 0.9, 'Finance': 1.2, 'Operations': 0.95
        }
        dept_factor = [dept_multiplier[dept] for dept in data['department']]
        
        # Education factor
        edu_multiplier = {
            'High School': 0.8, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.6
        }
        edu_factor = [edu_multiplier[edu] for edu in data['education_level']]
        
        # Job title factor
        title_multiplier = {
            'Junior': 0.7, 'Mid-level': 1.0, 'Senior': 1.4,
            'Lead': 1.6, 'Manager': 1.8, 'Director': 2.2
        }
        title_factor = [title_multiplier[title] for title in data['job_title']]
        
        # Performance factor
        perf_factor = data['performance_rating'] * 3000
        
        # Location factor
        loc_multiplier = {'Urban': 1.2, 'Suburban': 1.0, 'Rural': 0.85}
        loc_factor = [loc_multiplier[loc] for loc in data['location']]
        
        # Calculate salary with some random noise
        data['salary'] = (
            salary_base + 
            age_factor + 
            exp_factor + 
            perf_factor + 
            (data['overtime_hours'] * 500) +
            np.random.normal(0, 5000, n_samples)
        )
        
        # Apply multipliers
        data['salary'] = data['salary'] * np.array(dept_factor) * np.array(edu_factor) * np.array(title_factor) * np.array(loc_factor)
        
        # Ensure positive salaries
        data['salary'] = np.maximum(data['salary'], 25000)
        
        # Round salaries
        data['salary'] = np.round(data['salary'], 0)
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def load_data(self, filepath=None):
        """
        Load salary data from file or generate sample data
        
        Args:
            filepath (str): Path to CSV file, if None generates sample data
        """
        if filepath and pd.io.common.file_exists(filepath):
            self.data = pd.read_csv(filepath)
        else:
            self.data = self.generate_sample_data()
        
        print(f"Data loaded successfully! Shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        print("=== SALARY PREDICTION ANALYSIS ===\n")
        print("1. DATASET OVERVIEW")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\n2. DATA TYPES AND INFO")
        print(self.data.info())
        
        print("\n3. DESCRIPTIVE STATISTICS")
        print(self.data.describe())
        
        print("\n4. MISSING VALUES")
        print(self.data.isnull().sum())
        
        print("\n5. CATEGORICAL VARIABLES DISTRIBUTION")
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.data[col].value_counts())
    
    def visualize_data(self):
        """
        Create comprehensive visualizations of the salary data
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Salary Prediction Analysis - Comprehensive Data Visualization', fontsize=16, fontweight='bold')
        
        # 1. Salary distribution
        axes[0, 0].hist(self.data['salary'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Salary Distribution')
        axes[0, 0].set_xlabel('Salary ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Salary vs Experience
        axes[0, 1].scatter(self.data['years_experience'], self.data['salary'], alpha=0.6, color='green')
        axes[0, 1].set_title('Salary vs Years of Experience')
        axes[0, 1].set_xlabel('Years of Experience')
        axes[0, 1].set_ylabel('Salary ($)')
        
        # 3. Salary vs Age
        axes[0, 2].scatter(self.data['age'], self.data['salary'], alpha=0.6, color='red')
        axes[0, 2].set_title('Salary vs Age')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].set_ylabel('Salary ($)')
        
        # 4. Salary by Department
        dept_salary = self.data.groupby('department')['salary'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(dept_salary)), dept_salary.values, color='orange')
        axes[1, 0].set_title('Average Salary by Department')
        axes[1, 0].set_ylabel('Average Salary ($)')
        axes[1, 0].set_xticks(range(len(dept_salary)))
        axes[1, 0].set_xticklabels(dept_salary.index, rotation=45)
        
        # 5. Salary by Education Level
        edu_salary = self.data.groupby('education_level')['salary'].mean().sort_values(ascending=False)
        axes[1, 1].bar(range(len(edu_salary)), edu_salary.values, color='purple')
        axes[1, 1].set_title('Average Salary by Education Level')
        axes[1, 1].set_ylabel('Average Salary ($)')
        axes[1, 1].set_xticks(range(len(edu_salary)))
        axes[1, 1].set_xticklabels(edu_salary.index, rotation=45)
        
        # 6. Salary by Job Title
        title_salary = self.data.groupby('job_title')['salary'].mean().sort_values(ascending=False)
        axes[1, 2].bar(range(len(title_salary)), title_salary.values, color='brown')
        axes[1, 2].set_title('Average Salary by Job Title')
        axes[1, 2].set_ylabel('Average Salary ($)')
        axes[1, 2].set_xticks(range(len(title_salary)))
        axes[1, 2].set_xticklabels(title_salary.index, rotation=45)
        
        # 7. Performance vs Salary
        axes[2, 0].scatter(self.data['performance_rating'], self.data['salary'], alpha=0.6, color='pink')
        axes[2, 0].set_title('Salary vs Performance Rating')
        axes[2, 0].set_xlabel('Performance Rating')
        axes[2, 0].set_ylabel('Salary ($)')
        
        # 8. Overtime Hours vs Salary
        axes[2, 1].scatter(self.data['overtime_hours'], self.data['salary'], alpha=0.6, color='yellow')
        axes[2, 1].set_title('Salary vs Overtime Hours')
        axes[2, 1].set_xlabel('Overtime Hours')
        axes[2, 1].set_ylabel('Salary ($)')
        
        # 9. Correlation Heatmap
        # Prepare numerical data for correlation
        numerical_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numerical_data.corr()
        
        im = axes[2, 2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[2, 2].set_title('Correlation Matrix')
        axes[2, 2].set_xticks(range(len(correlation_matrix.columns)))
        axes[2, 2].set_yticks(range(len(correlation_matrix.columns)))
        axes[2, 2].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[2, 2].set_yticklabels(correlation_matrix.columns)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[2, 2])
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/Python-Learner/Python-Learner/salary_analysis_visualization.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create additional detailed visualizations
        self._create_detailed_visualizations()
    
    def _create_detailed_visualizations(self):
        """Create additional detailed visualizations"""
        
        # Box plots for categorical variables
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Salary Distribution by Categorical Variables', fontsize=14, fontweight='bold')
        
        # Department boxplot
        sns.boxplot(data=self.data, x='department', y='salary', ax=axes[0, 0])
        axes[0, 0].set_title('Salary by Department')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Education boxplot
        sns.boxplot(data=self.data, x='education_level', y='salary', ax=axes[0, 1])
        axes[0, 1].set_title('Salary by Education Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Job title boxplot
        sns.boxplot(data=self.data, x='job_title', y='salary', ax=axes[1, 0])
        axes[1, 0].set_title('Salary by Job Title')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Location boxplot
        sns.boxplot(data=self.data, x='location', y='salary', ax=axes[1, 1])
        axes[1, 1].set_title('Salary by Location')
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/Python-Learner/Python-Learner/salary_boxplots.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_data_for_modeling(self):
        """
        Prepare data for machine learning models
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Create a copy of the data for modeling
        model_data = self.data.copy()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['department', 'education_level', 'job_title', 'location']
        
        for col in categorical_cols:
            le = LabelEncoder()
            model_data[col + '_encoded'] = le.fit_transform(model_data[col])
            label_encoders[col] = le
        
        # Select features for modeling
        feature_cols = ['age', 'years_experience', 'performance_rating', 'overtime_hours',
                       'department_encoded', 'education_level_encoded', 'job_title_encoded', 'location_encoded']
        
        X = model_data[feature_cols]
        y = model_data['salary']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple machine learning models for salary prediction
        """
        if self.X_train is None:
            self.prepare_data_for_modeling()
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        # Train models and store results
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            # Store model and results
            self.models[name] = model
            self.results[name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'predictions': y_pred_test
            }
            
            print(f"{name} Results:")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MSE: {test_mse:.2f}")
            print(f"  Test MAE: {test_mae:.2f}")
    
    def evaluate_models(self):
        """
        Evaluate and compare model performance
        """
        if not self.results:
            raise ValueError("No models trained. Please train models first.")
        
        print("\n=== MODEL EVALUATION SUMMARY ===")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Test R²': results['test_r2'],
                'Test MSE': results['test_mse'],
                'Test MAE': results['test_mae']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Visualize model comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        
        models = list(self.results.keys())
        r2_scores = [self.results[model]['test_r2'] for model in models]
        mse_scores = [self.results[model]['test_mse'] for model in models]
        mae_scores = [self.results[model]['test_mae'] for model in models]
        
        # R² comparison
        axes[0].bar(models, r2_scores, color=['blue', 'green'])
        axes[0].set_title('R² Score (Higher is Better)')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)
        
        # MSE comparison
        axes[1].bar(models, mse_scores, color=['blue', 'green'])
        axes[1].set_title('Mean Squared Error (Lower is Better)')
        axes[1].set_ylabel('MSE')
        
        # MAE comparison
        axes[2].bar(models, mae_scores, color=['blue', 'green'])
        axes[2].set_title('Mean Absolute Error (Lower is Better)')
        axes[2].set_ylabel('MAE')
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/Python-Learner/Python-Learner/model_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot predictions vs actual
        self._plot_predictions_vs_actual()
    
    def _plot_predictions_vs_actual(self):
        """Plot predicted vs actual values for all models"""
        
        fig, axes = plt.subplots(1, len(self.models), figsize=(15, 5))
        if len(self.models) == 1:
            axes = [axes]
        
        fig.suptitle('Predicted vs Actual Salary', fontsize=14, fontweight='bold')
        
        for i, (model_name, results) in enumerate(self.results.items()):
            predictions = results['predictions']
            
            axes[i].scatter(self.y_test, predictions, alpha=0.6)
            axes[i].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Salary')
            axes[i].set_ylabel('Predicted Salary')
            axes[i].set_title(f'{model_name}\nR² = {results["test_r2"]:.3f}')
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/Python-Learner/Python-Learner/predictions_vs_actual.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance for Random Forest model
        """
        if 'Random Forest' not in self.models:
            print("Random Forest model not found. Please train models first.")
            return
        
        rf_model = self.models['Random Forest']
        feature_names = ['age', 'years_experience', 'performance_rating', 'overtime_hours',
                        'department', 'education_level', 'job_title', 'location']
        
        # Get feature importances
        importances = rf_model.feature_importances_
        
        # Create DataFrame for easier visualization
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        print(feature_importance_df.to_string(index=False))
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.title('Feature Importance - Random Forest Model')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('/home/runner/work/Python-Learner/Python-Learner/feature_importance.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_salary(self, age, years_experience, department, education_level, 
                      job_title, performance_rating, overtime_hours, location):
        """
        Predict salary for a new employee
        
        Args:
            age (int): Employee age
            years_experience (int): Years of experience
            department (str): Department name
            education_level (str): Education level
            job_title (str): Job title
            performance_rating (float): Performance rating (1-5)
            overtime_hours (int): Overtime hours per month
            location (str): Work location
            
        Returns:
            dict: Predictions from all trained models
        """
        if not self.models:
            raise ValueError("No models trained. Please train models first.")
        
        # Create input DataFrame (simplified encoding for demonstration)
        dept_encoding = {'Engineering': 2, 'Sales': 4, 'Marketing': 3, 'HR': 1, 'Finance': 0, 'Operations': 5}
        edu_encoding = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        title_encoding = {'Junior': 2, 'Mid-level': 3, 'Senior': 4, 'Lead': 1, 'Manager': 5, 'Director': 0}
        loc_encoding = {'Urban': 2, 'Suburban': 1, 'Rural': 0}
        
        input_data = pd.DataFrame({
            'age': [age],
            'years_experience': [years_experience],
            'performance_rating': [performance_rating],
            'overtime_hours': [overtime_hours],
            'department_encoded': [dept_encoding.get(department, 0)],
            'education_level_encoded': [edu_encoding.get(education_level, 0)],
            'job_title_encoded': [title_encoding.get(job_title, 0)],
            'location_encoded': [loc_encoding.get(location, 0)]
        })
        
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(input_data)[0]
            predictions[model_name] = round(pred, 2)
        
        print(f"\n=== SALARY PREDICTION ===")
        print(f"Employee Profile:")
        print(f"  Age: {age}")
        print(f"  Experience: {years_experience} years")
        print(f"  Department: {department}")
        print(f"  Education: {education_level}")
        print(f"  Job Title: {job_title}")
        print(f"  Performance: {performance_rating}/5")
        print(f"  Overtime Hours: {overtime_hours}")
        print(f"  Location: {location}")
        print(f"\nPredicted Salaries:")
        for model_name, pred_salary in predictions.items():
            print(f"  {model_name}: ${pred_salary:,.2f}")
        
        return predictions
    
    def run_complete_analysis(self):
        """
        Run the complete salary prediction analysis pipeline
        """
        print("Starting Comprehensive Salary Prediction Analysis...")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Visualize data
        print("\nCreating visualizations...")
        self.visualize_data()
        
        # Step 4: Prepare and train models
        print("\nPreparing data for modeling...")
        self.prepare_data_for_modeling()
        
        print("\nTraining machine learning models...")
        self.train_models()
        
        # Step 5: Evaluate models
        print("\nEvaluating model performance...")
        self.evaluate_models()
        
        # Step 6: Feature importance
        print("\nAnalyzing feature importance...")
        self.feature_importance_analysis()
        
        # Step 7: Example prediction
        print("\nExample prediction:")
        self.predict_salary(
            age=35,
            years_experience=8,
            department='Engineering',
            education_level='Master',
            job_title='Senior',
            performance_rating=4.2,
            overtime_hours=5,
            location='Urban'
        )
        
        print("\n" + "=" * 60)
        print("Analysis Complete! Check the generated visualization files:")
        print("- salary_analysis_visualization.png")
        print("- salary_boxplots.png") 
        print("- model_comparison.png")
        print("- predictions_vs_actual.png")
        print("- feature_importance.png")


def main():
    """
    Main function to run the salary prediction analysis
    """
    analyzer = SalaryAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()