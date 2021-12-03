import os
from DataGenerator import DataGenerator

def generate_training_data(fp, features, labels, spectators, bump_size=3):
	'''
	Using DataGenerator object implemented by the mentors,
	generate_data() loads and reads multiple gigabytes of data
	from `.root` files. DataGenerator will process and provide training data
	separated by multiple batches as providing all training data at once
	will not fit in typical memory space.

	Argument for `fp` parameter should be a filepath to `train` directory,
	or directory that contains all training files, not to one specific training file
	i.e argument MUST end with `/` character

	`features`, `labels`, `spectators` arguments intake list of names of features, labels, and
	spectator columns

	DataGenerator implementation is explained in `DataGenerator.py`

	Function will create multiple DataGenerator objects, each of which generates
	`bump_size` amount of training data, or three training files by default.
	This is in an effort to use `boosting` approach to train our Fully Connected NN classifier
	as fitting same NN classifier multiple times with different set of training data
	will achieve greater generalization of the classifier on any data.
	Using this approach actually enhanced the performance of our model by computing more than
	10% higher generated AUC.

	Thus, function will return list of multiple DataGenerators that generate training data

	NOTE: Test data will get generated directly using DataGenerator object
	'''
	assert fp[-1] == '/', '`fp` must be a filepath to directory, not file'

	contents = os.listdir(fp)
	training_files = [fp+x for x in contents]
	training_gens = list()
	counter=0

	for i in range(0, len(training_files), bump_size):
		training_temp = training_files[i:i+bump_size]
		training_generator = DataGenerator(training_temp, features, labels, spectators,
			remove_unlabeled=True, max_entry=8000)
		training_gens.append(training_generator)
		counter+=1;
		print(f'{counter} DataGenerator instantiated')

	return training_gens