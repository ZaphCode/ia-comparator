import os
import matplotlib.pyplot as plt

"""
Genera y guarda una gráfica de barras con los porcentajes de acierto
de cada clasificador.
"""
def plot_accuracies(acc_dict, out_path='results/accuracy_comparison.png'):
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    names = list(acc_dict.keys())
    values = [acc_dict[n] for n in names]

    plt.figure(figsize=(7,5))
    plt.bar(names, values)
    plt.ylim(0, 1.0)
    plt.ylabel('Porcentaje de acierto')
    plt.title('Comparación de accuracy entre clasificadores')

    # Mostrar valores arriba de las barras
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


"""
Genera y guarda la imagen de la matriz de confusión.
"""
def plot_confusion_matrix(cm, labels, filename):
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure(figsize=(5,4))
    plt.imshow(cm)
    plt.title('Matriz de confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor real')

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    # Escribir los valores dentro de la matriz
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()