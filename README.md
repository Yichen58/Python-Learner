# Python-Learner
An AI agent's code journal for learning Python for business analysis as a beginner

## 🎯 Salary Prediction Analysis Project

This repository contains a comprehensive salary prediction analysis project that demonstrates Python data science techniques for business applications.

### 📊 Project Overview

The **Salary Prediction Analysis** is a complete data science project that includes:
- Synthetic salary dataset generation with realistic relationships
- Comprehensive exploratory data analysis (EDA)
- Interactive data visualizations
- Multiple machine learning models for salary prediction
- Model evaluation and comparison
- Feature importance analysis
- Practical prediction interface

### 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Complete Analysis**
   ```bash
   python salary_prediction_analysis.py
   ```

3. **Run Tests**
   ```bash
   python test_salary_analysis.py
   ```

4. **Interactive Analysis**
   ```bash
   jupyter notebook salary_analysis_notebook.ipynb
   ```

### 📁 Project Structure

```
Python-Learner/
├── salary_prediction_analysis.py    # Main analysis module
├── salary_analysis_notebook.ipynb   # Interactive Jupyter notebook
├── test_salary_analysis.py         # Test script
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
└── Generated Outputs/
    ├── salary_analysis_visualization.png
    ├── salary_boxplots.png
    ├── model_comparison.png
    ├── predictions_vs_actual.png
    └── feature_importance.png
```

### 🔍 Features

#### Data Analysis
- **Synthetic Dataset**: Realistic salary data with 8 features and 1000+ samples
- **Data Exploration**: Comprehensive statistical analysis and data profiling
- **Missing Value Analysis**: Complete data quality assessment

#### Visualizations
- **Distribution Plots**: Salary distribution and relationships
- **Correlation Analysis**: Feature correlation heatmap
- **Category Analysis**: Salary by department, education, job title
- **Performance Metrics**: Model comparison charts

#### Machine Learning
- **Multiple Models**: Linear Regression and Random Forest
- **Model Evaluation**: R², MSE, MAE metrics
- **Feature Importance**: Identification of key salary drivers
- **Prediction Interface**: Easy-to-use salary prediction function

### 📈 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Employee age (22-65) |
| `years_experience` | Numeric | Years of work experience (0-40) |
| `department` | Categorical | Engineering, Sales, Marketing, HR, Finance, Operations |
| `education_level` | Categorical | High School, Bachelor, Master, PhD |
| `job_title` | Categorical | Junior, Mid-level, Senior, Lead, Manager, Director |
| `performance_rating` | Numeric | Performance score (1.0-5.0) |
| `overtime_hours` | Numeric | Monthly overtime hours (0-20) |
| `location` | Categorical | Urban, Suburban, Rural |
| `salary` | Numeric | Annual salary (target variable) |

### 🎓 Learning Objectives

This project demonstrates:

1. **Data Science Workflow**
   - Data generation and loading
   - Exploratory data analysis
   - Data preprocessing and feature engineering
   - Model training and evaluation

2. **Python Libraries**
   - `pandas` for data manipulation
   - `numpy` for numerical computations
   - `matplotlib` and `seaborn` for visualization
   - `scikit-learn` for machine learning

3. **Business Analysis Skills**
   - Understanding salary determinants
   - Statistical analysis and interpretation
   - Model selection and evaluation
   - Practical business recommendations

### 🔧 Usage Examples

#### Basic Usage
```python
from salary_prediction_analysis import SalaryAnalyzer

# Initialize analyzer
analyzer = SalaryAnalyzer()

# Run complete analysis
analyzer.run_complete_analysis()
```

#### Custom Prediction
```python
# Predict salary for a specific employee profile
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
```

#### Step-by-Step Analysis
```python
# Load and explore data
analyzer.load_data()
analyzer.explore_data()

# Create visualizations
analyzer.visualize_data()

# Train and evaluate models
analyzer.prepare_data_for_modeling()
analyzer.train_models()
analyzer.evaluate_models()
```

### 📊 Sample Results

**Model Performance:**
- Linear Regression: R² ≈ 0.85-0.90
- Random Forest: R² ≈ 0.90-0.95

**Key Insights:**
- Job title and education level are strongest salary predictors
- Years of experience shows strong positive correlation
- Department choice significantly impacts earning potential
- Performance rating directly affects compensation

### 🎯 Business Applications

1. **HR Analytics**
   - Salary benchmarking and market analysis
   - Compensation planning and budgeting
   - Pay equity analysis

2. **Career Planning**
   - Understanding salary progression paths
   - Identifying high-impact career decisions
   - Education ROI analysis

3. **Recruitment**
   - Competitive salary offer determination
   - Talent acquisition cost estimation
   - Market positioning analysis

### 🧪 Testing

The project includes comprehensive tests:
- Functionality validation
- Data quality checks
- Model performance verification
- Error handling validation

Run tests with:
```bash
python test_salary_analysis.py
```

### 📚 Dependencies

- Python 3.7+
- pandas 2.1.4
- numpy 1.24.3
- matplotlib 3.7.2
- seaborn 0.12.2
- scikit-learn 1.3.2
- jupyter 1.0.0

### 🔄 Future Enhancements

- [ ] Real-world dataset integration
- [ ] Advanced feature engineering
- [ ] Deep learning models
- [ ] Interactive web dashboard
- [ ] Time series salary analysis
- [ ] Geographic salary mapping

### 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

### 🤝 Contributing

This is a learning repository. Feel free to:
- Submit issues for bugs or improvements
- Create pull requests for enhancements
- Share your own analysis variations
- Suggest additional features

### 📞 Contact

For questions or suggestions about this project, please open an issue in the repository.

---

*This project is part of a Python learning journey focused on business analysis and data science applications.*
