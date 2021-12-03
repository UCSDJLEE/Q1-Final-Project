from sklearn.metrics import roc_curve, auc

def compute_metrics(true_labels, preds):
	'''
	Compute metrics -- FPR, TPR, and AUC -- to evaluate the performance of our classifier
	These three metrics will be used to demonstrate roc_curve in our `visualization.ipynb`
	notebook under `notebooks` directory
	'''
	fpr, tpr, threshold = roc_curve(true_labels, preds)
	auc_val = auc(fpr, tpr)

	return fpr, tpr, auc_val