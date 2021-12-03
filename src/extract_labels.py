import yaml

def extract_labels(fp):
	'''
	The names of features, target columns, and more are 
	listed and stored in `.yml` file

	extract_feature_labels() will use Python `yaml` library
	to read and convert the contents in our `.yml` file into Python dictionary
	
	Function will return all values associated to every key 
	in (feature_labels, spectator_labels, target_labels) followed by their length in respective order
	'''

	with open(fp, 'r') as f:
		defs = yaml.load(fp, Loader=yaml.FullLoader) # FullLoader converts data from yml structure to Python dictionary

	features = defs['features']
	spectators = defs['spectators']
	labels = defs['labels']

	nfeatures = defs['nfeatures']
	nspectators = defs['nspectators']
	nlabels = defs['nlabels']

	return (features, spectators, labels, nfeatures, nspectators, nlabels)