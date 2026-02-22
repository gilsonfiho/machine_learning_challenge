from modules.project_3.data_loader import DataLoader
from modules.project_3.model import ModelBuilder
from modules.project_3.trainer import Trainer
from modules.project_3.views import Views
from modules.project_3.config import Config

def main():
    print("="*50)
    print("CLASSIFICADOR DE TIPOS DE CARROS")
    print("="*50)
    
    # 1. Carregar dados
    print("\n1. Carregando dados...")
    classes = Config.CLASSES
    num_classes = len(classes) + 1
    
    X, y = DataLoader.load_images_from_folder(
        f"{Config.DATASET_PATH}/bmw10_ims",
        classes
    )
    print(f"   Total de imagens: {len(X)}")
    print(f"   Classes: {classes} + {Config.OTHER_CLASS}")
    
    # 2. Normalizar e dividir
    print("\n2. Dividindo dados...")
    X = DataLoader.normalize(X)
    X_train, X_val, X_test, y_train, y_val, y_test = DataLoader.split_with_mat(
        X, y, 
        "data/bmw10_release/bmw10_annos.mat"
    )
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. Construir modelo
    print("\n3. Construindo modelo...")
    model = ModelBuilder.build_model(num_classes)
    model = ModelBuilder.compile_model(model)
    print(f"   Modelo pronto: {num_classes} classes")
    
    # 4. Treinar
    print("\n4. Treinando...")
    history = Trainer.train(model, X_train, y_train, X_val, y_val)
    
    # 5. Avaliar
    print("\n5. Avaliando...")
    loss, accuracy = Trainer.evaluate(model, X_test, y_test)
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    
    # 6. Predições e métricas
    print("\n6. Gerando resultados...")
    y_pred = Trainer.predict(model, X_test)
    
    class_names = [str(c) for c in classes] + [Config.OTHER_CLASS]
    Views.plot_history(history, class_names)
    Views.plot_confusion_matrix(y_test, y_pred, class_names)
    Views.print_metrics(y_test, y_pred, class_names)
    
    print("\n" + "="*50)
    print("TREINAMENTO CONCLUÍDO!")
    print(f"Modelo salvo em: {Config.MODEL_PATH}")
    print(f"Resultados em: {Config.RESULTS_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()