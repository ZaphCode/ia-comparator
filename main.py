import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import plot_accuracies, plot_confusion_matrix
from models import run_knn, run_naive_bayes, run_svm

"""
Carga y prepara los datos para los modelos.
- Lee el archivo CSV usando codificación latin1 (para manejar acentos)
- Convierte la columna 'Azúcares' de formato '0,8' a 0.8 si existe
- Codifica la columna 'Marca' como números (LabelEncoder) si existe
- Elimina filas con valores faltantes
- Separa y regresa X (características), y (clases) y el dataframe completo
"""
def load_and_preprocess(path):
    df = pd.read_csv(path, encoding='latin1')
    print("Dataset cargado. Dimensiones:", df.shape)

    # Convertir Azúcares a número flotante
    if 'Azúcares' in df.columns:
        df['Azucares'] = df['Azucares'].astype(str).str.replace(',', '.')
        df['Azucares'] = pd.to_numeric(df['Azucares'], errors='coerce')

    # Remover filas con valores faltantes
    df = df.dropna().reset_index(drop=True)

    # Codificar Marca en números
    if 'Marca' in df.columns:
        le = LabelEncoder()
        df['Marca_enc'] = le.fit_transform(df['Marca'].astype(str))

    # Columna objetivo
    target_col = 'Categoria'
    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_col}' en el dataset.")

    # Seleccionar solo columnas numéricas como características
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    X = df[numeric_cols].copy()
    y = df[target_col].astype(int).copy()

    return X, y, df


def main():
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Cargar y preprocesar datos
    X, y, df = load_and_preprocess('cookies-data.csv')
    print("Características utilizadas:", list(X.columns))
    print("\nDistribución de clases:")
    print(y.value_counts())

    # Separar datos en entrenamiento/prueba manteniendo proporciones
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Estandarizar los datos (muy importante para KNN y SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar clasificadores
    knn_model, knn_acc, knn_report, knn_cm = run_knn(
        X_train, X_test, y_train, y_test, n_neighbors=5
    )
    nb_model, nb_acc, nb_report, nb_cm = run_naive_bayes(
        X_train, X_test, y_train, y_test
    )
    svm_model, svm_acc, svm_report, svm_cm = run_svm(
        X_train, X_test, y_train, y_test, C=1.0, kernel='rbf'
    )

    accs = {'KNN': knn_acc, 'NaiveBayes': nb_acc, 'SVM': svm_acc}

    # Mostrar resultados en terminal
    print("\n=== PORCENTAJES DE ACIERTO ===")
    for name, a in accs.items():
        print(f"{name}: {a:.4f}")

    print("\n=== Reporte KNN ===\n", knn_report)
    print("\n=== Reporte Naive Bayes ===\n", nb_report)
    print("\n=== Reporte SVM ===\n", svm_report)

    # Graficar comparaciones
    plot_accuracies(accs, out_path=os.path.join(results_dir, 'accuracy_comparison.png'))

    # Etiquetas de clases para las matrices de confusión
    labels = sorted(list(map(int, sorted(y.unique()))))

    plot_confusion_matrix(knn_cm, labels, os.path.join(results_dir, 'knn_confusion.png'))
    plot_confusion_matrix(nb_cm, labels, os.path.join(results_dir, 'nb_confusion.png'))
    plot_confusion_matrix(svm_cm, labels, os.path.join(results_dir, 'svm_confusion.png'))

    # Crear resumen CSV
    summary = pd.DataFrame([{'Algoritmo': k, 'Precisión': v} for k, v in accs.items()])
    summary.to_csv(os.path.join(results_dir, 'accuracies_summary.csv'), index=False)

    print("Archivos generados:")
    
    for fname in [
        'accuracy_comparison.png',
        'knn_confusion.png',
        'nb_confusion.png',
        'svm_confusion.png',
        'accuracies_summary.csv'
    ]:
        print('-', os.path.join(results_dir, fname))


if __name__ == '__main__':
    main()