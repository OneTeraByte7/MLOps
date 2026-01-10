import sys
sys.path.append('src')

from src.models.train import ChurnModelTrainer

if __name__ == "__main__":
    print("Starting model training...")
    trainer = ChurnModelTrainer()
    trainer.run_training_pipeline()
    print("\nTraining completed!")
