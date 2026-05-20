import matplotlib
import matplotlib.pyplot as plt

class Drawer:
    matplotlib.rcParams["figure.figsize"] = (10, 8)

    @staticmethod
    def plot_history(
        train_losses: list[float],
        train_accuracies: list[float],
        test_losses: list[float],
        test_accuracies: list[float],
        save_path: str | None = None
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5))
        
        ax1.plot(train_losses, "blue", linewidth = 2, label = "Train Loss")
        if test_losses:
            ax1.plot(test_losses, "red", linewidth = 2, label = "Test Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Test Loss")
        ax1.grid(True, alpha = 0.3)
        ax1.legend()

        ax2.plot(train_accuracies, "green", linewidth = 2, label = "Train Accuracy")
        if test_accuracies:
            ax2.plot(test_accuracies, "orange", linewidth = 2, label = "Test Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Test Accuracy")
        ax2.grid(True, alpha = 0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
