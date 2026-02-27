import torch
from tqdm import tqdm
from src.training.scheduler import EarlyStopping
import os
import matplotlib.pyplot as plt


def train_model(model, model_name, trainloader, valloader, criterion, optimizer,
                scheduler, device, num_epochs, early_stopping_patience, 
                fine_tuning, save_dir):
    """
    Treina o modelo com Early Stopping e LR Scheduler
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    model.to(device)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    best_model_state = None
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # ================== TREINO ==================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(trainloader, desc="  Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss = train_loss / len(trainloader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # ================== VALIDAÇÃO ==================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(valloader, desc="  Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / len(valloader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Logging
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # ================== SCHEDULER ==================
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"  LR: {old_lr:.2e} → {new_lr:.2e}")
        
        # ================== EARLY STOPPING ==================
        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping na epoch {epoch+1}")
            break
        
        # ================== SAVE BEST ==================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Restaurar melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Melhor modelo restaurado (Val Loss: {best_val_loss:.4f})")
    
    # Salvar modelo
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))
        
        # Plotar história
        plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                            model_name, save_dir)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }


def plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                         model_name, save_dir):
    """Plota história de treinamento"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss por Época')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy por Época')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), dpi=300)
    plt.close()
