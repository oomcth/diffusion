import torch


def save(model: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         scheduler: torch.optim.lr_scheduler._LRScheduler,
         current_epoch: int,
         filepath: str):
    if not filepath.endswith('.pth'):
        filepath = filepath + '.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': current_epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint sauvegardé à {filepath}")


def load(model: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         scheduler: torch.optim.lr_scheduler._LRScheduler,
         filepath: str):
    if not filepath.endswith('.pth'):
        filepath = filepath + '.pth'
    filepath = "train/checkpoint2.pth"  # ou le chemin de ton fichier

    with open(filepath, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    current_epoch = checkpoint['epoch']
    return model, optimizer, scheduler, current_epoch
