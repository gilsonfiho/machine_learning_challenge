"""
src/evaluation/metrics.py - Métricas e avaliação
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    auc,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
import os


def evaluate_model(model, dataloader, device, class_names, save_dir=None):
    """
    Avaliação completa do modelo
    
    Args:
        model: Modelo PyTorch
        dataloader: DataLoader de teste
        device: 'cuda' ou 'cpu'
        class_names: Nomes das classes
        save_dir: Diretório para salvar resultados
        
    Returns:
        dict: Dicionário com métricas
    """
    import torch
    from tqdm import tqdm
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calcular métricas
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    print("\n" + "="*70)
    print("MÉTRICAS DE AVALIAÇÃO")
    print("="*70)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plotar ROC
    if save_dir:
        plot_roc_curve(all_labels, all_probs, class_names, save_dir)
        plot_confusion_matrix(all_labels, all_preds, class_names, save_dir)
        plot_metrics_per_class(precision, recall, f1, class_names, save_dir)
    
    return {
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
    }


def plot_roc_curve(y_true, y_probs, class_names, save_dir):
    """Plota curva ROC multiclasse"""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_probs = np.array(y_probs)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='navy', lw=2, linestyle='--',
             label=f'Macro-avg (AUC = {macro_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


def plot_metrics_per_class(precision, recall, f1, class_names, save_dir):
    """Plota métricas por classe"""
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    plt.xticks(x, class_names)
    plt.ylabel('Score')
    plt.title('Métricas por Classe')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_per_class.png'), dpi=300)
    plt.close()
