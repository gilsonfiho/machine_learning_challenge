import os
import sys
import numpy as np

sys.path.insert(0, '.')

from modules.project_3.data_loader import DataLoader
from modules.project_3.tuner import HyperparameterTuner
from modules.project_3.config import Config

def main():
    print("="*50)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS (OneDrive Safe Mode)")
    print("="*50)
    
    dataset_path = f"{Config.DATASET_PATH}/bmw10_ims"
    mat_path = "data/bmw10_release/bmw10_annos.mat"
    
    # Pasta de resultados final (dentro do seu projeto)
    final_output_folder = os.path.abspath("hiper_results")
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder, exist_ok=True)
    
    # 1. Carregar dados
    print("\n1. Carregando dados...")
    classes = Config.CLASSES
    X, y = DataLoader.load_images_from_folder(dataset_path, classes)
    
    # 2. Normalizar e dividir
    print("\n2. Dividindo dados...")
    X = DataLoader.normalize(X)
    X_train, X_val, X_test, y_train, y_val, y_test = DataLoader.split_with_mat(X, y, mat_path)
    
    # 3. Escolher método
    print("\n3. Escolhendo método...")
    print("   1. Random Search")
    print("   2. Bayesian Optimization")
    choice = input("   Escolha (1 ou 2): ").strip()
    
    if choice == "1":
        tuner = HyperparameterTuner.tune_random(X_train, y_train, X_val, y_val, max_trials=5)
    else:
        tuner = HyperparameterTuner.tune_bayesian(X_train, y_train, X_val, y_val, max_trials=5)
    
    # 4. Exibir e Salvar Resultados
    HyperparameterTuner.print_results(tuner)
    
    print("\n5. Salvando melhores parâmetros no projeto...")
    _, best_hps = HyperparameterTuner.get_best_model(tuner)
    
    res_file = os.path.join(final_output_folder, 'best_hyperparameters.txt')
    with open(res_file, 'w') as f:
        f.write("MELHORES HIPERPARÂMETROS ENCONTRADOS\n")
        f.write("="*40 + "\n")
        for p in ['learning_rate', 'dense_1', 'dense_2', 'dropout_1', 'dropout_2', 'weight_decay']:
            f.write(f"{p}: {best_hps.get(p)}\n")
            
    print(f"   Sucesso! Melhores parâmetros salvos em: {res_file}")
    print("   Os logs temporários foram processados em C:/temp para evitar erros de permissão.")

if __name__ == "__main__":
    main()