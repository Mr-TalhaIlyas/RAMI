from termcolor import cprint
import torch, os
from config import config
import numpy as  np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

def save_chkpt(model, optimizer, epoch=0, loss=0, acc=0, return_chkpt=False):
    cprint('-> Saving checkpoint', 'green')
    torch.save({
                'epoch': epoch,
                'loss': loss,
                'acc': acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}.pth'))#_epoch{epoch}
    cprint(os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}.pth'), 'cyan')#_epoch{epoch}
    if return_chkpt:
        return os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}.pth')#_epoch{epoch}
    

def values_fromreport(report):
    p = report['weighted avg']['precision']
    r = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    return p,r, f1

def get_num_segments(w,t,o):
    '''
    Parameters
    ----------
    w : widnow length default, 10sec.
    t : Time in sec covered by n-segments.
    o : overlap in sec between consecutive windows.
    Returns
    -------
    n : number of consecutve seconds required to cover t seconds.
    '''
    n = ((t-w)/(w-o)) + 1
    return int(np.ceil(n))

def plot_tsne(embeddings, labels, legends=True, exclude_class_0=False):
    # sns.set(style="whitegrid")

    # Filter out class 0 if required
    if exclude_class_0:
        mask = labels != 0
        embeddings = embeddings[mask]
        labels = labels[mask]
        # Recalculate unique classes after filtering
        classes = np.unique(labels)
        # Map labels to a continuous range starting from 0
        label_to_id = {label: id for id, label in enumerate(classes)}
        mapped_labels = np.array([label_to_id[label] for label in labels])
    else:
        classes = np.unique(labels)
        mapped_labels = labels  # Use original labels if not excluding class 0

    # Create a scatter plot
    # plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=mapped_labels, cmap='Set1', s=3)
    plt.gca().set_aspect('equal', 'datalim')

    # Create a colorbar with correct ticks and labels
    cbar = plt.colorbar(boundaries=np.arange(len(classes)+1)-0.5, ticks=np.arange(len(classes)))
    cbar.set_ticklabels(classes)
    
    # Optionally add legends
    if legends:
        # Create custom legends with class labels
        legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
        plt.gca().add_artist(legend1)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    plt.tight_layout()  # Adjust layout to not cut off elements
    plt.show()
    # plt.axis('off')

def plot_tsne_3d(embeddings, labels):
    fig = plt.figure()#figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                         c=labels,
                        cmap='tab10_r',
                        s=1)
    plt.legend(*scatter.legend_elements(), title='Classes')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.tight_layout()
    plt.show()

def calculate_confusion_matrix_gtcs_pnes(lbls, preds, filenames, tolerance=1):

    d = {ni: indi for indi, ni in enumerate(set(filenames))}
    pat_ids = [d[ni] for ni in filenames]
    # Initialize dictionaries to store labels and predictions by patient ID
    patient_labels = defaultdict(list)
    patient_predictions = defaultdict(list)
    
    # Aggregate labels and predictions by patient
    for lbl, pred, pat_id in zip(lbls, preds, pat_ids):
        patient_labels[pat_id].append(lbl)
        patient_predictions[pat_id].append(pred)
    
    # Initialize the confusion matrix counts
    TP = TN = FP = FN = 0

    # Process each patient's data
    for pat_id in patient_labels:
        labels = np.array(patient_labels[pat_id])
        predictions = np.array(patient_predictions[pat_id])

        # Count occurrences
        lbl_count_0 = np.sum(labels == 0)
        lbl_count_1 = np.sum(labels == 1)
        pred_count_0 = np.sum(predictions == 0)
        pred_count_1 = np.sum(predictions == 1)

        # Determine the majority class in labels and predictions
        majority_lbl = 0 if lbl_count_0 > lbl_count_1 else 1
        majority_pred = 0 if pred_count_0 > pred_count_1 else 1
        
        # Check misclassifications (allowance for configurable tolerance)
        misclassified = np.sum(labels != predictions)
        
        # Apply the rules for TP, TN, FP, FN with configured tolerance
        if majority_lbl == majority_pred:
            if misclassified <= tolerance:
                if majority_lbl == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if majority_lbl == 1:
                    FN += 1
                else:
                    FP += 1
        else:
            if majority_lbl == 1:
                FN += 1
            else:
                FP += 1

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

def plot_cm(cm, cls_names=['+ve', '-ve']):
    plt.figure(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False,
                annot_kws={"size": 14}, square=True)  # Adjust font size and remove color bar

    # plt.xlabel('Predicted label', fontsize=14)
    # plt.ylabel('True label', fontsize=14)
    plt.xticks(ticks=[0.5, 1.5], labels=cls_names, fontsize=12)
    plt.yticks(ticks=[0.5, 1.5], labels=cls_names, fontsize=12, rotation=0)

    plt.tight_layout()
    plt.show()

    return None

def plot_cm2(cm, metrics=[0,0,0,0], tol=0, cls_names=['+ve', '-ve']):

    # Pre-calculated values
    sensitivity = metrics[0]
    precision = metrics[1]
    specificity = metrics[2]
    f1_score = metrics[3]

    # Metrics matrix for heatmap
    metrics_matrix = np.array([
        [sensitivity, precision],
        [specificity, f1_score]
    ])
    metrics_labels = [['Sen/Rec', 'Pre'], ['Spe', 'F1']]

    # Plot the heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(3, 2))
    # seagreen  viridis Blues
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 10}, square=True, ax=axs[0])  # Adjust font size and remove color bar
    axs[0].set_xticks([0.5, 1.5])
    axs[0].set_xticklabels(cls_names, fontsize=9)
    axs[0].set_yticks([0.5, 1.5])
    axs[0].set_yticklabels(cls_names, fontsize=9, rotation=0)
    # axs[0].set_title('Confusion Matrix')

    sns.heatmap(metrics_matrix, annot=True, fmt='.2f', cmap=sns.light_palette("seagreen", as_cmap=True),
                cbar=False,
                annot_kws={"size": 10}, square=True, ax=axs[1])
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(['', f'Tolerance={tol}'], fontsize=8)
    axs[1].set_yticks([]) #turn off y-axis ticks
    # axs[1].set_yticklabels(['Metric 1', 'Metric 2'], fontsize=12, rotation=0)
    # axs[1].set_title('Metrics')
    # axs[1].axis('off')
    # Custom labels for the metrics heatmap
    for y in range(metrics_matrix.shape[0]):
        for x in range(metrics_matrix.shape[1]):
            axs[1].text(x + 0.5, y + 0.18, f'{metrics_labels[y][x]}', 
                        horizontalalignment='center', verticalalignment='center', color='k', fontsize=8)

    plt.tight_layout()
    plt.show()

    return None


#%%
# same as calculate_confusion_matrix_gtcs_pnes

def findMaxConsecutive(nums, which=1):
    max_count = 0
    count = 0
    for i in nums:
        if i == which:
            count += 1
        else:
            max_count = max(max_count, count)
            count = 0
    return max(max_count, count)

def evaluate_gtcs_pnes(all_lbls, all_preds, files, threshold=3, ratio_thresh=0.25):

    threshold = threshold
    ratio_thresh = ratio_thresh # 25% of wrong alarms
    TP = TN = FP = FN = 0
    
    lbls = all_lbls
    preds = all_preds
    filenames = files
    d = {ni: indi for indi, ni in enumerate(set(filenames))}
    pat_ids = [d[ni] for ni in filenames]
    # Initialize dictionaries to store labels and predictions by patient ID
    patient_labels = defaultdict(list)
    patient_predictions = defaultdict(list)

    # Aggregate labels and predictions by patient
    for lbl, pred, pat_id in zip(lbls, preds, pat_ids):
        patient_labels[pat_id].append(lbl)
        patient_predictions[pat_id].append(pred)

    gts = list(patient_labels.values())
    preds = list(patient_predictions.values())

    for i in range(len(gts)):
    
        lbl = list(set(gts[i]))[0]
        _, counts = np.unique(preds[i], return_counts=True)
        
        if lbl == 0: # -ve class TN
            max_counts = findMaxConsecutive(preds[i], 1)
            try:
                ratio = counts[1] / counts[0]
            except IndexError:
                ratio = 0
            if max_counts <= threshold and ratio <= ratio_thresh:
                TN += 1
            else:
                FP += 1
            
            
        if lbl == 1: # -ve class TP
            max_counts = findMaxConsecutive(preds[i], 0)
            try:
                ratio = counts[0] / counts[1]
            except IndexError:
                ratio = 0
            if max_counts <= threshold and ratio <= ratio_thresh:
                TP += 1
            else:
                FN += 1
            
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
