import os
from utils.data import Data
from net.net import CVNet
from net.train_test import TrainerTester
from utils.drawer import Drawer
from config import *

def main():
    os.makedirs("models", exist_ok = True)
    os.makedirs("plots", exist_ok = True)
    
    print("Загрузка данных")
    data = Data()
    data.load()
    train_loader, test_loader = data.get_loaders(batch_size = BATCH_SIZE)

    model = CVNet(hidden_neurons = HIDDEN_NEURONS)
    print(f"Модель создана с {HIDDEN_NEURONS} нейронами в скрытом слое")

    print("Производится обучение и тестирование")
    trainer_tester = TrainerTester(model)
    history = trainer_tester.train_test(
        criterion = CRITERION,
        train_dataloader = train_loader,
        epochs = EPOCHS,
        lr = LEARNING_RATE,
        optimizer_class = OPTIMIZER,
        test_dataloader = test_loader,
        **OPTIMIZER_KWARGS
    )
    if trainer_tester.save(MODEL_PATH):
        print(f"Модель сохранена в {MODEL_PATH}")
    else:
        print("Не удалось сохранить модель")

    Drawer.plot_history(
        train_losses = history["train_loss"],
        train_accuracies = history["train_acc"],
        test_losses = history["test_loss"],
        test_accuracies = history["test_acc"],
        save_path = "plots/training_history.png"
    )

    print(f"\nИтоговые метрики")
    print(f"Train Loss: {history['train_loss'][-1]:.2f}")
    print(f"Train Acc: {history['train_acc'][-1]:.2f}")
    print(f"Test Loss: {history['test_loss'][-1]:.2f}")
    print(f"Test Acc: {history['test_acc'][-1]:.2f}")

if __name__ == "__main__":
    main()
