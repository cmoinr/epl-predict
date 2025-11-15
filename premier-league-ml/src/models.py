"""
Módulo de Modelos ML para Premier League
Contiene Random Forest y Gradient Boosting para predicción de resultados y goles
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class EPLModelTrainer:
    """Entrena y evalúa modelos para predicción de resultados y goles"""
    
    def __init__(self, X_train, X_test, y_result_train, y_result_test, y_goals_train, y_goals_test):
        """
        Inicializa con datos de train/test
        
        Args:
            X_train, X_test: Features
            y_result_train, y_result_test: Target resultado (1X2)
            y_goals_train, y_goals_test: Target goles totales
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_result_train = y_result_train
        self.y_result_test = y_result_test
        self.y_goals_train = y_goals_train
        self.y_goals_test = y_goals_test
        
        self.models = {}
        self.results = {}
    
    def train_result_models(self) -> Dict:
        """
        Entrena modelos para predicción de resultado (1X2)
        
        Returns:
            Dict con resultados de entrenamiento
        """
        print("\n" + "="*70)
        print("ENTRENAMIENTO - PREDICCIÓN DE RESULTADOS (1X2)")
        print("="*70)
        
        results = {}
        
        # 1. Random Forest Classifier
        print("\n1️⃣  Random Forest Classifier")
        print("   Entrenando...")
        
        rf_result = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_result.fit(self.X_train, self.y_result_train)
        
        y_pred_rf = rf_result.predict(self.X_test)
        y_pred_proba_rf = rf_result.predict_proba(self.X_test)
        
        rf_acc = accuracy_score(self.y_result_test, y_pred_rf)
        rf_precision = precision_score(self.y_result_test, y_pred_rf, average='weighted')
        rf_recall = recall_score(self.y_result_test, y_pred_rf, average='weighted')
        rf_f1 = f1_score(self.y_result_test, y_pred_rf, average='weighted')
        
        try:
            rf_auc = roc_auc_score(self.y_result_test, y_pred_proba_rf, multi_class='ovr', average='weighted')
        except:
            rf_auc = 0
        
        print(f"   ✅ Accuracy:  {rf_acc:.4f} ({rf_acc*100:.2f}%)")
        print(f"   ✅ Precision: {rf_precision:.4f}")
        print(f"   ✅ Recall:    {rf_recall:.4f}")
        print(f"   ✅ F1-Score:  {rf_f1:.4f}")
        print(f"   ✅ ROC-AUC:   {rf_auc:.4f}")
        
        results['Random Forest'] = {
            'model': rf_result,
            'predictions': y_pred_rf,
            'accuracy': rf_acc,
            'precision': rf_precision,
            'recall': rf_recall,
            'f1': rf_f1,
            'auc': rf_auc
        }
        
        # 2. Gradient Boosting Classifier
        print("\n2️⃣  Gradient Boosting Classifier")
        print("   Entrenando...")
        
        gb_result = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_result.fit(self.X_train, self.y_result_train)
        
        y_pred_gb = gb_result.predict(self.X_test)
        y_pred_proba_gb = gb_result.predict_proba(self.X_test)
        
        gb_acc = accuracy_score(self.y_result_test, y_pred_gb)
        gb_precision = precision_score(self.y_result_test, y_pred_gb, average='weighted')
        gb_recall = recall_score(self.y_result_test, y_pred_gb, average='weighted')
        gb_f1 = f1_score(self.y_result_test, y_pred_gb, average='weighted')
        
        try:
            gb_auc = roc_auc_score(self.y_result_test, y_pred_proba_gb, multi_class='ovr', average='weighted')
        except:
            gb_auc = 0
        
        print(f"   ✅ Accuracy:  {gb_acc:.4f} ({gb_acc*100:.2f}%)")
        print(f"   ✅ Precision: {gb_precision:.4f}")
        print(f"   ✅ Recall:    {gb_recall:.4f}")
        print(f"   ✅ F1-Score:  {gb_f1:.4f}")
        print(f"   ✅ ROC-AUC:   {gb_auc:.4f}")
        
        results['Gradient Boosting'] = {
            'model': gb_result,
            'predictions': y_pred_gb,
            'accuracy': gb_acc,
            'precision': gb_precision,
            'recall': gb_recall,
            'f1': gb_f1,
            'auc': gb_auc
        }
        
        # Comparar
        print("\n" + "-"*70)
        print("COMPARACIÓN:")
        print("-"*70)
        
        if gb_acc > rf_acc:
            print(f"🏆 Mejor: Gradient Boosting ({gb_acc:.4f} vs {rf_acc:.4f})")
        else:
            print(f"🏆 Mejor: Random Forest ({rf_acc:.4f} vs {gb_acc:.4f})")
        
        self.models['result'] = results
        return results
    
    def train_goals_models(self) -> Dict:
        """
        Entrena modelos para predicción de goles totales
        
        Returns:
            Dict con resultados de entrenamiento
        """
        print("\n" + "="*70)
        print("ENTRENAMIENTO - PREDICCIÓN DE GOLES TOTALES")
        print("="*70)
        
        results = {}
        
        # 1. Random Forest Regressor
        print("\n1️⃣  Random Forest Regressor")
        print("   Entrenando...")
        
        rf_goals = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_goals.fit(self.X_train, self.y_goals_train)
        
        y_pred_rf = rf_goals.predict(self.X_test)
        
        rf_mae = mean_absolute_error(self.y_goals_test, y_pred_rf)
        rf_rmse = np.sqrt(mean_squared_error(self.y_goals_test, y_pred_rf))
        rf_r2 = r2_score(self.y_goals_test, y_pred_rf)
        
        print(f"   ✅ MAE:  {rf_mae:.4f}")
        print(f"   ✅ RMSE: {rf_rmse:.4f}")
        print(f"   ✅ R²:   {rf_r2:.4f}")
        
        results['Random Forest'] = {
            'model': rf_goals,
            'predictions': y_pred_rf,
            'mae': rf_mae,
            'rmse': rf_rmse,
            'r2': rf_r2
        }
        
        # 2. Gradient Boosting Regressor
        print("\n2️⃣  Gradient Boosting Regressor")
        print("   Entrenando...")
        
        gb_goals = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_goals.fit(self.X_train, self.y_goals_train)
        
        y_pred_gb = gb_goals.predict(self.X_test)
        
        gb_mae = mean_absolute_error(self.y_goals_test, y_pred_gb)
        gb_rmse = np.sqrt(mean_squared_error(self.y_goals_test, y_pred_gb))
        gb_r2 = r2_score(self.y_goals_test, y_pred_gb)
        
        print(f"   ✅ MAE:  {gb_mae:.4f}")
        print(f"   ✅ RMSE: {gb_rmse:.4f}")
        print(f"   ✅ R²:   {gb_r2:.4f}")
        
        results['Gradient Boosting'] = {
            'model': gb_goals,
            'predictions': y_pred_gb,
            'mae': gb_mae,
            'rmse': gb_rmse,
            'r2': gb_r2
        }
        
        # Comparar
        print("\n" + "-"*70)
        print("COMPARACIÓN:")
        print("-"*70)
        
        if gb_r2 > rf_r2:
            print(f"🏆 Mejor: Gradient Boosting (R²: {gb_r2:.4f} vs {rf_r2:.4f})")
        else:
            print(f"🏆 Mejor: Random Forest (R²: {rf_r2:.4f} vs {gb_r2:.4f})")
        
        self.models['goals'] = results
        return results
    
    def feature_importance(self, model_type: str = 'result', model_name: str = 'Gradient Boosting') -> pd.DataFrame:
        """
        Retorna importancia de features
        
        Args:
            model_type: 'result' o 'goals'
            model_name: 'Random Forest' o 'Gradient Boosting'
        
        Returns:
            DataFrame con importancia
        """
        model = self.models[model_type][model_name]['model']
        feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def print_summary(self):
        """Imprime resumen de todos los modelos"""
        print("\n" + "="*70)
        print("RESUMEN FINAL DE MODELOS")
        print("="*70)
        
        if 'result' in self.models:
            print("\n📊 PREDICCIÓN DE RESULTADOS (1X2):")
            for name, metrics in self.models['result'].items():
                print(f"\n  {name}:")
                print(f"    • Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    • F1-Score:  {metrics['f1']:.4f}")
                print(f"    • ROC-AUC:   {metrics['auc']:.4f}")
        
        if 'goals' in self.models:
            print("\n📊 PREDICCIÓN DE GOLES TOTALES:")
            for name, metrics in self.models['goals'].items():
                print(f"\n  {name}:")
                print(f"    • MAE:  {metrics['mae']:.4f}")
                print(f"    • RMSE: {metrics['rmse']:.4f}")
                print(f"    • R²:   {metrics['r2']:.4f}")


if __name__ == '__main__':
    print("🤖 Módulo de Modelos ML - EPL Premier League")
    print("\nUso:")
    print("  1. from src.models import EPLModelTrainer")
    print("  2. trainer = EPLModelTrainer(X_train, X_test, y_result_train, y_result_test, y_goals_train, y_goals_test)")
    print("  3. trainer.train_result_models()")
    print("  4. trainer.train_goals_models()")
