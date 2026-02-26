"""
src/visualizations/plotter.py - Funções de visualização e plotagem
Movido de visualization/plotter.py para organização profissional
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns


def plot_class_distribution(datasets_dict, save_dir):
    """
    Plota distribuição de classes
    
    Args:
        datasets_dict: {'train': dataset, 'val': dataset, 'test': dataset}
        save_dir: Diretório para salvar
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, dataset) in enumerate(datasets_dict.items()):
        labels = [sample[1] for sample in dataset.samples] if hasattr(dataset, 'samples') else []
        if not labels:
            continue
        
        counts = Counter(labels)
        classes = sorted(counts.keys())
        values = [counts[c] for c in classes]
        
        axes[idx].bar(classes, values, color='skyblue', edgecolor='navy')
        axes[idx].set_title(f'Distribuição {name.upper()}')
        axes[idx].set_xlabel('Classe')
        axes[idx].set_ylabel('Número de Amostras')
        axes[idx].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plota matriz de confusão
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Nomes das classes
        save_path: Caminho para salvar
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, num_classes, class_names, save_path):
    """
    Plota curva ROC para cada classe
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas
        num_classes: Número de classes
        class_names: Nomes das classes
        save_path: Caminho para salvar
    """
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_per_class(precision, recall, f1, class_names, save_path):
    """
    Plota métricas por classe
    
    Args:
        precision: Precisão por classe
        recall: Recall por classe
        f1: F1-score por classe
        class_names: Nomes das classes
        save_path: Caminho para salvar
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision')
    bars2 = ax.bar(x, recall, width, label='Recall')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score')
    
    ax.set_ylabel('Score')
    ax.set_title('Metrics per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(reports_dict, save_dir):
    """
    Plota resumo de métricas dos modelos
    
    Args:
        reports_dict: {'model_name': metrics_dict, ...}
        save_dir: Diretório para salvar
    """
    models = list(reports_dict.keys())
    f1_scores = [reports_dict[m].get('weighted avg', {}).get('f1-score', 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_ylabel('F1-Score (Weighted)')
    ax.set_title('Comparação de Modelos')
    ax.set_ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'models_comparison.png'), dpi=300)
    plt.close()


def plot_training_history(history, model_name, save_path):
    """
    Plota histórico de treinamento (loss e acurácia)
    
    Args:
        history: Dict com 'train_losses', 'val_losses', 'train_accs', 'val_accs'
        model_name: Nome do modelo
        save_path: Caminho para salvar
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_losses'], label='Train Loss', marker='o')
    ax1.plot(history['val_losses'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training History (Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_accs'], label='Train Acc', marker='o')
    ax2.plot(history['val_accs'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Training History (Accuracy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
