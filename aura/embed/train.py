from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from ..dataset import DatasetProvider
from .model import EmotionModel


def create_heatmap(true, pred, path):
    cm = confusion_matrix(true, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if path:
        plt.savefig(f'{path}/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    provider = DatasetProvider()

    print("Training Data Distr.")
    provider.label_distr()
    print("=" * 15, "\n")

    print("Testing Data Distr.")
    provider.label_distr(test = True)
    print("=" * 15, "\n")

    model = EmotionModel(pretrained = True)

    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model.train_loop(provider = provider, epochs = 3, batch_size = 32)
    model.test(provider)
    
    print("Saving model...")
    model.save()