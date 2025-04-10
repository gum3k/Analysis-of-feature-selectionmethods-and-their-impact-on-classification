import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import time
import matplotlib.pyplot as plt

# Funkcja do ewaluacji klasyfikatora przy użyciu walidacji krzyżowej
def evaluate_model_cv(X, y, model, n_splits_list):
    scores_dict = {}
    for n_splits in n_splits_list:
        skf = StratifiedKFold(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        scores_dict[n_splits] = scores
    return scores_dict

# Funkcja do usuwania cech stałych
def remove_constant_features(X, feature_names):
    selector = VarianceThreshold(threshold=0.0)
    X_new = selector.fit_transform(X)
    retained_features = selector.get_support(indices=True)
    return X_new, [feature_names[i] for i in retained_features]

# Funkcja do imputacji brakujących wartości
def impute_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(X)

# Funkcja do analizy wybranych cech
def analyze_selected_features(feature_names, selected_indices):
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features

# Funkcja do pomiaru czasu wykonania
def measure_execution_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Wczytanie zestawów danych
mice_protein = pd.read_csv('mice-protein.csv')
wine_quality = pd.read_csv('winequality-red.csv')
breast_cancer = pd.read_csv('breast_cancer.csv')



# Dostosowanie zestawów danych do odpowiednich kolumn docelowych
datasets = {
    'Mice Protein': (mice_protein.drop(columns=['MouseID', 'Genotype', 'Treatment', 'Behavior', 'class']), mice_protein['class']),
    'Wine Quality': (wine_quality.drop(columns=['quality']), wine_quality['quality']),
    'Breast Cancer': (breast_cancer.drop(columns=['id', 'Unnamed: 32', 'diagnosis']), breast_cancer['diagnosis']),
}

# Funkcja do usuwania cech o wysokiej korelacji
def remove_high_corr_features(X, threshold=0.8):
    cor_matrix = X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return X.drop(columns=to_drop), to_drop

# Funkcja do ewaluacji metod selekcji cech na różnych zestawach danych
def evaluate_feature_selection_methods(datasets, n_splits_list):
    results_rf = {}
    feature_analysis = {}
    execution_times = {}
    scaler = StandardScaler()

    for name, (X, y) in datasets.items():
        # Przechowywanie oryginalnych nazw cech
        feature_names = X.columns.tolist()

        # Konwersja etykiet na wartości numeryczne, jeśli nie są już numeryczne
        if y.dtype == 'O' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Imputacja brakujących wartości
        X = pd.DataFrame(impute_missing_values(X), columns=feature_names)

        # Usuwanie cech stałych
        X, feature_names = remove_constant_features(X, feature_names)

        # Ponowna imputacja brakujących wartości po usunięciu cech stałych
        X = pd.DataFrame(impute_missing_values(X), columns=feature_names)

        # Skalowanie danych
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

        execution_times[name] = {}

        # Selekcja cech metodą Variance Threshold
        X_var, execution_time_var = measure_execution_time(VarianceThreshold(threshold=0.1).fit_transform, X)
        selected_features_var = VarianceThreshold(threshold=0.1).fit(X).get_support(indices=True)
        execution_times[name]['Variance Threshold'] = execution_time_var

        # Selekcja cech na podstawie korelacji
        X_corr, high_corr_features = remove_high_corr_features(X)
        selected_features_corr = [i for i in range(len(feature_names)) if feature_names[i] not in high_corr_features]
        execution_times[name]['Correlation-based'] = measure_execution_time(lambda: X.drop(columns=high_corr_features))[1]

        # Ważność cech według Random Forest
        clf_rf = RandomForestClassifier(n_estimators=100)
        clf_rf.fit(X, y)
        importances = clf_rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        important_features = indices[:5]
        X_imp = X.iloc[:, important_features]
        execution_times[name]['Feature Importance'] = measure_execution_time(clf_rf.fit, X, y)[1]

        # Recursive Feature Elimination
        n_features_to_select = min(5, X.shape[1])
        rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=n_features_to_select, step=1)
        X_rfe, execution_time_rfe = measure_execution_time(rfe_selector.fit_transform, X, y)
        selected_features_rfe = np.where(rfe_selector.support_)[0]
        execution_times[name]['RFE'] = execution_time_rfe

        # SelectKBest z f_classif
        kbest_selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
        X_kbest, execution_time_kbest = measure_execution_time(kbest_selector.fit_transform, X, y)
        selected_features_kbest = kbest_selector.get_support(indices=True)
        execution_times[name]['SelectKBest'] = execution_time_kbest

        # Ewaluacja modelu przy użyciu walidacji krzyżowej dla różnych wartości k
        scores_dict = {}
        for k in n_splits_list:
            scores_dict[k] = {
                'Variance Threshold': evaluate_model_cv(X_var, y, RandomForestClassifier(n_estimators=100), [k]),
                'Correlation-based': evaluate_model_cv(X_corr, y, RandomForestClassifier(n_estimators=100), [k]),
                'Feature Importance': evaluate_model_cv(X_imp, y, RandomForestClassifier(n_estimators=100), [k]),
                'RFE': evaluate_model_cv(X_rfe, y, RandomForestClassifier(n_estimators=100), [k]),
                'SelectKBest': evaluate_model_cv(X_kbest, y, RandomForestClassifier(n_estimators=100), [k])
            }

        results_rf[name] = scores_dict

        feature_analysis[name] = {
            'Variance Threshold': analyze_selected_features(feature_names, selected_features_var),
            'Correlation-based': analyze_selected_features(feature_names, selected_features_corr),
            'Feature Importance': analyze_selected_features(feature_names, important_features),
            'RFE': analyze_selected_features(feature_names, selected_features_rfe),
            'SelectKBest': analyze_selected_features(feature_names, selected_features_kbest)
        }

    return results_rf, feature_analysis, execution_times

# Lista wartości k dla walidacji krzyżowej
n_splits_list = [3, 5, 7, 10]

# Ewaluacja metod selekcji cech
comparison_results_rf, feature_analysis, execution_times = evaluate_feature_selection_methods(datasets, n_splits_list)

# Wyciągnięcie średnich wyników dla każdej metody selekcji cech
average_results = {}
for dataset, results in comparison_results_rf.items():
    average_results[dataset] = {}
    for k, methods in results.items():
        for method, scores in methods.items():
            if method not in average_results[dataset]:
                average_results[dataset][method] = []
            average_results[dataset][method].append(np.mean(scores[k]))

# Przedstawienie średnich wyników na wykresie
for dataset, methods in average_results.items():
    plt.figure(figsize=(10, 6))
    for method, averages in methods.items():
        plt.plot(n_splits_list, averages, marker='o', label=method)
    plt.title(f'Średnie wyniki dokładności dla {dataset}')
    plt.xlabel('Liczba podziałów (k)')
    plt.ylabel('Średnia dokładność')
    plt.legend()
    plt.grid(True)
    plt.show()
# Wyświetlenie średnich wyników
print("\nŚrednie wyniki dokładności dla każdej metody selekcji cech:")
for dataset, methods in average_results.items():
    print(f"\nZestaw danych: {dataset}")
    for method, averages in methods.items():
        print(f"{method}: {averages}")

# Wyświetlenie wyników
print("Wyniki dokładności Random Forest przy użyciu walidacji krzyżowej:")
for dataset, results in comparison_results_rf.items():
    print(f"\nZestaw danych: {dataset}")
    for k, methods in results.items():
        print(f"\nLiczba podziałów: {k}")
        for method, scores in methods.items():
            print(f"{method}: {scores}")

print("\nAnaliza selekcji cech:")
for dataset, methods in feature_analysis.items():
    print(f"\nSelekcja cech dla {dataset}:")
    for method, features in methods.items():
        print(f"{method}: {features}")

print("\nCzasy wykonania metod selekcji cech:")
for dataset, times in execution_times.items():
    print(f"\nCzasy wykonania dla {dataset}:")
    for method, time in times.items():
        print(f"{method}: {time} sekund")
