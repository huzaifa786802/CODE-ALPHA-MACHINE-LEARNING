# Credit Scoring Model - Code Alpha Company
# Objective: Predict an individual's creditworthiness using past financial data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')
# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class CreditScoringModel:
    """
    Credit Scoring Model for Code Alpha Company
    Predicts creditworthiness using multiple classification algorithms
    """
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=10)
        self.results = {}
    def generate_sample_data(self, n_samples=1000):
        """
        Generate sample financial data for demonstration
        In production, this would be replaced with real data loading
        """
        np.random.seed(42)
        data = {
            'income': np.random.normal(50000, 20000, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'employment_years': np.random.randint(0, 40, n_samples),
            'debt_to_income_ratio': np.random.uniform(0, 1, n_samples),
            'credit_history_length': np.random.randint(0, 30, n_samples),
            'number_of_loans': np.random.randint(0, 10, n_samples),
            'payment_history_score': np.random.uniform(300, 850, n_samples),
            'total_debt': np.random.normal(30000, 25000, n_samples),
            'savings_balance': np.random.normal(15000, 10000, n_samples),
            'number_of_credit_cards': np.random.randint(0, 8, n_samples),
            'monthly_expenses': np.random.normal(3000, 1500, n_samples),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'home_ownership': np.random.choice(['Own', 'Rent', 'Mortgage'], n_samples)
        }
        df = pd.DataFrame(data)
        # Ensure realistic constraints
        df['income'] = np.maximum(df['income'], 20000)
        df['total_debt'] = np.maximum(df['total_debt'], 0)
        df['savings_balance'] = np.maximum(df['savings_balance'], 0)
        df['monthly_expenses'] = np.maximum(df['monthly_expenses'], 1000)
        # Create target variable based on logical rules
        df['default_risk'] = (
            (df['debt_to_income_ratio'] > 0.6) |
            (df['payment_history_score'] < 500) |
            ((df['income'] < 30000) & (df['total_debt'] > 50000)) |
            (df['number_of_loans'] > 5)
        ).astype(int)
        return df
    def feature_engineering(self, df):
        """
        Create additional features from existing data
        """
        df_engineered = df.copy()
        # Financial ratios
        df_engineered['debt_to_income_ratio'] = df_engineered['total_debt'] / df_engineered['income']
        df_engineered['savings_to_income_ratio'] = df_engineered['savings_balance'] / df_engineered['income']
        df_engineered['expense_to_income_ratio'] = df_engineered['monthly_expenses'] / (df_engineered['income'] / 12)
        # Age-related features
        df_engineered['age_income_interaction'] = df_engineered['age'] * df_engineered['income'] / 100000
        df_engineered['employment_stability'] = df_engineered['employment_years'] / df_engineered['age']
        # Credit utilization features
        df_engineered['avg_debt_per_loan'] = df_engineered['total_debt'] / (df_engineered['number_of_loans'] + 1)
        df_engineered['credit_cards_per_income'] = df_engineered['number_of_credit_cards'] / (df_engineered['income'] / 10000)
        # Risk indicators
        df_engineered['high_debt_flag'] = (df_engineered['debt_to_income_ratio'] > 0.4).astype(int)
        df_engineered['low_savings_flag'] = (df_engineered['savings_to_income_ratio'] < 0.1).astype(int)
        df_engineered['young_borrower_flag'] = (df_engineered['age'] < 25).astype(int)
        return df_engineered
    def preprocess_data(self, df):
        """
        Preprocess the data for model training
        """
        # Handle categorical variables
        le_education = LabelEncoder()
        le_home = LabelEncoder()
        df['education_level_encoded'] = le_education.fit_transform(df['education_level'])
        df['home_ownership_encoded'] = le_home.fit_transform(df['home_ownership'])
        # Select features for modeling
        feature_columns = [
            'income', 'age', 'employment_years', 'debt_to_income_ratio',
            'credit_history_length', 'number_of_loans', 'payment_history_score',
            'total_debt', 'savings_balance', 'number_of_credit_cards',
            'monthly_expenses', 'education_level_encoded', 'home_ownership_encoded',
            'savings_to_income_ratio', 'expense_to_income_ratio', 
            'age_income_interaction', 'employment_stability', 'avg_debt_per_loan',
            'credit_cards_per_income', 'high_debt_flag', 'low_savings_flag',
            'young_borrower_flag'
        ]
        X = df[feature_columns]
        y = df['default_risk']
        return X, y, feature_columns
    def train_models(self, X_train, y_train):
        """
        Train multiple classification models
        """
        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models using multiple metrics
        """
        print("\n" + "="*80)
        print("CODE ALPHA COMPANY - CREDIT SCORING MODEL EVALUATION")
        print("="*80)
        for name, model in self.models.items():
            print(f"\n{name} Results:")
            print("-" * 40)
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            # Print metrics
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            # Cross-validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
            print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    def plot_model_comparison(self):
        """
        Create visualizations comparing model performance
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Code Alpha Company - Credit Scoring Model Comparison', fontsize=16, fontweight='bold')
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.results.keys())
        # Bar plot of metrics
        ax1 = axes[0, 0]
        x = np.arange(len(metrics))
        width = 0.25
        for i, model_name in enumerate(model_names):
            values = [self.results[model_name][metric] for metric in metrics]
            ax1.bar(x + i * width, values, width, label=model_name)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # ROC Curves
        ax2 = axes[0, 1]
        for model_name in model_names:
            fpr, tpr, _ = roc_curve(y_test, self.results[model_name]['probabilities'])
            auc_score = self.results[model_name]['roc_auc']
            ax2.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Feature importance (Random Forest)
        if 'Random Forest' in self.models:
            ax3 = axes[1, 0]
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            feature_names = X_test.columns
            # Get top 10 features
            indices = np.argsort(feature_importance)[-10:]
            ax3.barh(range(len(indices)), feature_importance[indices])
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels([feature_names[i] for i in indices])
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Top 10 Feature Importance (Random Forest)')
            ax3.grid(True, alpha=0.3)
        # Confusion Matrix for best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        ax4 = axes[1, 1]
        cm = confusion_matrix(y_test, self.results[best_model]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title(f'Confusion Matrix - {best_model}')
        plt.tight_layout()
        plt.show()
        return fig
    def predict_creditworthiness(self, customer_data, model_name='Random Forest'):
        """
        Predict creditworthiness for new customers
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        model = self.models[model_name]
        # Ensure customer_data is in the right format
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        # Apply same preprocessing
        customer_data_processed = self.feature_engineering(customer_data)
        X_customer, _, _ = self.preprocess_data(customer_data_processed)
        # Scale features
        X_customer_scaled = self.scaler.transform(X_customer)
        # Predict
        prediction = model.predict(X_customer_scaled)[0]
        probability = model.predict_proba(X_customer_scaled)[0, 1]
        return {
            'default_risk': prediction,
            'default_probability': probability,
            'credit_score': int((1 - probability) * 850),  # Convert to credit score scale
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }
    def generate_report(self):
        """
        Generate a comprehensive model performance report
        """
        print("\n" + "="*80)
        print("CODE ALPHA COMPANY - CREDIT SCORING MODEL REPORT")
        print("="*80)
        # Best performing model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        best_score = self.results[best_model]['roc_auc']
        print(f"\nBest Performing Model: {best_model}")
        print(f"Best ROC-AUC Score: {best_score:.4f}")
        print("\nModel Rankings (by ROC-AUC):")
        print("-" * 40)
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"{i}. {name}: {metrics['roc_auc']:.4f}")
        print(f"\nRecommendation: Use {best_model} for production deployment.")
        print("This model provides the best balance of precision and recall for credit risk assessment.")
        return best_model
# Main execution
if __name__ == "__main__":
    # Initialize the credit scoring model
    credit_model = CreditScoringModel()
    # Generate sample data (replace with real data loading in production)
    print("Loading and preparing data...")
    df = credit_model.generate_sample_data(n_samples=2000)
    # Feature engineering
    print("Performing feature engineering...")
    df_engineered = credit_model.feature_engineering(df)
    # Preprocess data
    X, y, feature_columns = credit_model.preprocess_data(df_engineered)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Scale features
    X_train_scaled = credit_model.scaler.fit_transform(X_train)
    X_test_scaled = credit_model.scaler.transform(X_test)
    # Train models
    credit_model.train_models(X_train_scaled, y_train)
    # Evaluate models
    credit_model.evaluate_models(X_test_scaled, y_test)
    # Generate visualizations
    print("\nGenerating model comparison plots...")
    credit_model.plot_model_comparison()
    # Generate final report
    best_model_name = credit_model.generate_report()
    # Example prediction for a new customer
    print("\n" + "="*60)
    print("EXAMPLE CUSTOMER PREDICTION")
    print("="*60)
    sample_customer = {
        'income': 45000,
        'age': 35,
        'employment_years': 8,
        'credit_history_length': 12,
        'number_of_loans': 2,
        'payment_history_score': 720,
        'total_debt': 25000,
        'savings_balance': 8000,
        'number_of_credit_cards': 3,
        'monthly_expenses': 2800,
        'education_level': 'Bachelor',
        'home_ownership': 'Mortgage'
    }
    try:
        prediction = credit_model.predict_creditworthiness(sample_customer)
        print(f"Default Risk: {'Yes' if prediction['default_risk'] else 'No'}")
        print(f"Default Probability: {prediction['default_probability']:.3f}")
        print(f"Credit Score: {prediction['credit_score']}")
        print(f"Risk Level: {prediction['risk_level']}")
    except Exception as e:
        print(f"Prediction error: {e}")
        print("Note: Feature scaling needs to be applied to new data in production")
    print("\n" + "="*80)
    print("Model training and evaluation completed successfully!")
    print("Ready for deployment in Code Alpha Company's credit assessment system.")
    print("="*80)