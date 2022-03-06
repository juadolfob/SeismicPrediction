import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class PlotHelper:

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_cm(labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')



    def plot_metrics(history):

        metrics = ['loss', 'prc', 'Precision', 'Recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.tight_layout()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color=PlotHelper.colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_' + metric],
                     color=PlotHelper.colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])
            plt.legend()

    def plot_prc(name, labels, predictions, **kwargs):
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

        plt.plot(precision, recall, label=name, linewidth=2, color=PlotHelper.colors[0], **kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')

    def plot_roc(name, labels, predictions, **kwargs):
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

        plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
