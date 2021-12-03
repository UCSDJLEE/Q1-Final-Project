from tensorflow.keras.model import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(training_gens):
	'''
	Train the model using DataGenerator objects instantiated with training data.
	
	All DataGenerator will have consistent number of ntracks and nfeatures
	because of the way DataGenerator is implemented.

	Model will utilize Fully Connected Neural Network implemented using
	Functional API as it offers more flexibility than Sequential API 
	from tensorflow.keras open-source library.
	'''
	sample_generator = training_gens[0]
	sample_X = sample_generator[1][0]
	sample_y = sample_generator[1][1]
	ntracks, nfeatures = sample_X.shape[1:] # Note that sample_X.shape[0] represents number of jets per batch 
	nlabels = sample_y.shape[1]

	# Structuring Fully Connected NN
	inputs = Input(shape=(ntracks,nfeatures), name='InputLayer')
	x = BatchNormalization(name='BatchNorm')(inputs)
	x = Flatten(name='Flatten')(x)
	x = Dense(64, name='HiddenLayer1', activation='relu')(x)
	x = Dense(64, name='HiddenLayer2', activation='relu')(x) 
	x = Dense(32, name='HiddenLayer4', activation='relu')(x)
	outputs = Dense(nlabels, name='OutputLayer', activation='softmax')(x) # Binary classificatio
	nn_classifier = Model(inputs=inputs, outputs=outputs)
	# print(nn_classifier.summary())

	# Compile the classifier with appropriate `optimizer`, `loss`, `metrics` parameters

	# Optimzer:
	# Used Adam optimizer, a stochastic gradient descent method that is based on 
	# adaptive estimation of first-order and second-order moments

	# Loss:
	# Used binary cross entropy as we're performing binary classification(QCD or Hbb)

	# Evaluation metric:
	# Used accuracy to measure performance of classifer
	adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
	binary_loss = tf.keras.BinaryCrossentropy()
	binary_metric = tf.keras.metrics.BinaryAccuracy()

	nn_classifer.compile(optimizer=adam_optimizer,
					loss=binary_loss,
					metrics=[binary_metric])

	# Iterate through each DataGenerator in `training_gens`, fit the classifier
	# with certain callbacks for optimized learning
	early_stopper = EarlyStopping(monitor='val_loss', mode='min', patience=15)
	lr_reducer = ReduceLROnPlateau(patience=5, factor=0.5)
	callbacks = [early_stopper, lr_reducer]

	for gen in training_gens:
	    training_X, training_y = gen[1]
	    nn_classifier.fit(training_X, training_y, batch_size=1024, 
	                      epochs=100, validation_split=0.25, shuffle=False,
	                      callbacks=callbacks, verbose=0)

	return nn_classifer
