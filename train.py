from torch.utils.data import DataLoader
from trainers.trainer import Trainer
from utils.runtime import (
    prepare_runtime,
    build_mixed_dataset,
    build_model,
    create_data_splits,
    export_split_manifests,
)

def main():
    # 1. Load config
    config = prepare_runtime()
    print(f"Using device: {config['train']['device']}")
        
    # 2. Load dataset (Mixed scenes and sync errors)
    print("Building mixed dataset...")
    dataset = build_mixed_dataset(config)
    
    # 3. Split dataset
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset, config)
    print(
        f"Dataset split -> train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    export_split_manifests(train_dataset, val_dataset, test_dataset, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    
    # 4. Initialize model
    model = build_model(config)
    
    # 5. Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # 6. Start training
    trainer.fit()

if __name__ == "__main__":
    main()
