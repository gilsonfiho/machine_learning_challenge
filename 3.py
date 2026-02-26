import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import CONFIG, DEVICE
from src.data import load_bmw10_data, BMWDataset
from src.models.builder import get_model
from src.evaluation.metrics import evaluate_model
from src.training.scheduler import get_scheduler
from src.trainer import train_model
from src.visualizations.plotter import plot_class_distribution, plot_metrics_summary


def set_seed(seed=42):
    """Fixa seeds para reproducibilidade"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    """Retorna transformações de imagem"""
    basic = transforms.Compose([
        transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    augmented = transforms.Compose([
        transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return basic, augmented


def main():
    parser = argparse.ArgumentParser(description="BMW Classifier")
    parser.add_argument('--model', default='resnet50',
                        choices=['resnet50', 'efficientnet', 'convnext'])
    parser.add_argument('--all-models', action='store_true',
                        help='Treinar todos os modelos')
    parser.add_argument('--fine-tuning', action='store_true',
                        help='Usar fine-tuning')
    parser.add_argument('--batch-size', type=int, default=CONFIG['BATCH_SIZE'])

    args = parser.parse_args()

    # Setup
    set_seed(CONFIG['SEED'])

    print("\n" + "="*70)
    print("🚗 BMW CLASSIFIER - TRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Mode: {'Fine-tuning' if args.fine_tuning else 'Transfer Learning'}")
    print("="*70 + "\n")

    # Carregar dados
    print("[1/4] Carregando dados...")
    train_df, val_df, test_df = load_bmw10_data(
        CONFIG['DATA_DIR'], CONFIG['ANNOS_FILE'])

    basic_transform, augmented_transform = get_transforms()

    train_dataset = BMWDataset(
        train_df, CONFIG['IMG_DIR'],
        transform=augmented_transform,
        augment=True, augment_factor=CONFIG['AUGMENT_FACTOR']
    )
    val_dataset = BMWDataset(
        val_df, CONFIG['IMG_DIR'], transform=basic_transform)
    test_dataset = BMWDataset(
        test_df, CONFIG['IMG_DIR'], transform=basic_transform)

    trainloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=CONFIG['NUM_WORKERS'], pin_memory=CONFIG['PIN_MEMORY']
    )
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=CONFIG['NUM_WORKERS'], pin_memory=CONFIG['PIN_MEMORY'])
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=CONFIG['NUM_WORKERS'], pin_memory=CONFIG['PIN_MEMORY'])

    print(
        f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Plot distribuição
    plot_class_distribution(
        {'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
        CONFIG['FIGURES_DIR']
    )

    # Modelos
    print("\n[2/4] Inicializando modelos...")
    models_list = [args.model] if not args.all_models else [
        'resnet50', 'efficientnet', 'convnext']
    models_dict = {}
    for model_name in models_list:
        model = get_model(model_name, fine_tuning=args.fine_tuning)
        model.to(DEVICE)
        models_dict[model_name] = model
        print(f"  ✓ {model_name}")

    # Class weights
    print("\n[3/4] Preparando treinamento...")
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['class_final']),
        y=train_df['class_final']
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Treinar
    print("\n[4/4] Treinando...\n")
    all_reports = {}

    for model_name in models_list:
        print("="*70)
        print(f"MODELO: {model_name.upper()}")
        print("="*70)

        model = models_dict[model_name]

        lr = CONFIG['LR_FT'] if args.fine_tuning else CONFIG['LR_TL']
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=CONFIG['WEIGHT_DECAY']
        )
        scheduler = get_scheduler(
            optimizer,
            factor=CONFIG['LR_SCHEDULER_FACTOR'],
            patience=CONFIG['LR_SCHEDULER_PATIENCE'],
            min_lr=CONFIG['LR_SCHEDULER_MIN_LR']
        )

        save_dir = CONFIG['MODELS_DIR']
        os.makedirs(save_dir, exist_ok=True)

        model, history = train_model(
            model, model_name, trainloader, valloader, criterion, optimizer,
            scheduler, DEVICE, CONFIG['NUM_EPOCHS'],
            CONFIG['EARLY_STOPPING_PATIENCE'], args.fine_tuning, save_dir
        )

        # Avaliar
        print(f"\n→ Avaliando {model_name}...")
        eval_dir = os.path.join(CONFIG['FIGURES_DIR'], model_name)
        os.makedirs(eval_dir, exist_ok=True)

        metrics = evaluate_model(
            model, testloader, DEVICE, CONFIG['CLASS_NAMES'], eval_dir)
        all_reports[model_name] = metrics
        print()

    print("="*70)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("="*70)
    print(f"Modelos: {CONFIG['MODELS_DIR']}")
    print(f"Resultados: {CONFIG['FIGURES_DIR']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
