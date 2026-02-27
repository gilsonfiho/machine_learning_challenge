import torch
from tqdm import tqdm
from src.training.scheduler import EarlyStopping
import os
import matplotlib.pyplot as plt


def train_model(model, model_name, trainloader, valloader, criterion, optimizer,
                scheduler, device, num_epochs, early_stopping_patience, 
                fine_tuning, save_dir):
    
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
        
        # ============ TRAINING ============
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        with tqdm(trainloader, desc="Training") as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix(loss=loss.item())
        
        train_loss /= len(trainloader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # ============ VALIDATION ============
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            with tqdm(valloader, desc="Validation") as pbar:
                for images, labels in pbar:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    pbar.set_postfix(loss=loss.item())
        
        val_loss /= len(valloader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # ============ SCHEDULER & EARLY STOPPING ============
        if scheduler is not None:
            scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            # Salvar melhor modelo
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_model_state, os.path.join(save_dir, f'{model_name}_best.pth'))
            print(f"✓ Melhor modelo salvo!")
        
        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping acionado após {epoch+1} épocas")
            break
    
    # ============ PLOTAR HISTÓRIA ============
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - Training History (Loss)')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{model_name} - Training History (Accuracy)')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), dpi=300)
    plt.close()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss
    }
