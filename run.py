import os
import sys
import numpy as np

sys.path.insert(0, 'src') # Path specification for importing those utilities from our library codes

from extract_labels import extract_labels
from generate_training_data import generate_training_data
from train_model import train_model
from compute_metrics import compute_metrics
from DataGenerator import DataGenerator

def main():
	# First extract the names of attributes we need
	def_file = 'references/definitions.yml'
	features, spectators, labels, nfeatures, nspectators, nlabels = extract_labels(def_file)

	# Pass all training data files to DataGenerator to portion-batch process data
	# into our classifier for training
	training_dir_path = '/home/h8lee/teams/DSC180A_FA21_A00/a11/train/'
	training_gens = generate_training_data(training_dir_path, features, labels, spectators)

	# Train & generate model
	clf = train_model(training_gens)

	# Load test data into DataGenerator
	test_dir_path = '/home/h8lee/teams/DSC180A_FA21_A00/a11/test/'
	test_files = os.listdir(test_dir_path)
	test_file_path = [test_dir_path+f for f in test_files]
	test_generator = DataGenerator(test_file_path, features, labels, spectators,
					remove_unlabeled=True)

	# Make classifications on test data
	test_lbls = []
	test_preds = []

	for test in test_generator:
		truth_label = test[1]
		pred = clf.predict(test[0])

		test_lbls.append(truth_label)
		test_preds.append(pred)

	# Flatten out the dimensions of both truth labels and predictions
	test_lbls = np.concatenate(test_lbls, axis=0)
	test_preds = np.concatenate(test_preds, axis=0)

	# Extract `label_Hbb` column from both truth labels and predictions
	# as `label_Hbb` is our target label, and it retains binary relationship with
	# the other column, `label_QCD`. 
	# i.e when `label_QCD`=1 for a jet, `label_Hbb`=0 for that same jet.
	test_lbls = test_lbls[:,1]
	test_preds = test_preds[:,1]

	# Now compute all metrics to measure performance of our classifier
	fpr, tpr, auc = compute_metrics(test_lbls, test_preds)

	print(f'AUC = {round(auc*100, 2)}%')
	return

if __name__ == '__main__':
	main()