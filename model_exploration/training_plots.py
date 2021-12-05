import matplotlib.pyplot as plt
import os


def create_training_plot(history, output_path):
    assert "train_acc" in history.keys()
    assert "val_acc" in history.keys()

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1]);
    plt.savefig(os.path.join(output_path ,"training_plot.png"))

    return plt