import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from modules.project_3.config import Config

class Views:
    @staticmethod
    def plot_history(history, class_names):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Val')
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid()
        
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Val')
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid()
        
        plt.tight_layout()
        plt.savefig(f"{Config.RESULTS_PATH}/training_history.png")
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"{Config.RESULTS_PATH}/confusion_matrix.png")
        plt.close()
    
    @staticmethod
    def print_metrics(y_true, y_pred, class_names):
        print("\n" + "="*50)
        print("MÉTRICAS DE DESEMPENHO")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=class_names))