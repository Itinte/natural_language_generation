Natural Language Generation using deep learning models

__authors__ = [Louis Veillon, Quentin Boutoille-Blois]
__emails__ = [louis.veillon@essec.edu, b00527749@essec.edu]


### ALGORITHM PRESENTATION ###

Meaning Representation to Natural Language algorithm. We choose to implement a seq2seq model. It works as a encoder, decoder on either a character level or word level. In order to improve learning of the model we computed a lexicalization work applied beforehand to the dataset.

Our best model is already trained and saved (weights in 'saved_models/64_100_40k.h5', with its prediction in results/64_100_40k_results).
The model works on a character level.
The predictions are quite impressive and the seq2seq manages to understand perfectly the logical relation between the sentence Meaning Representation and the Natural Language.

It was trained on CPU for roughly 10 hours with the following hyperparameters:
-batch size: 64
-100 epochs
-latent dimension: 100
-cross validation split: 0.2
-categories to delexicalize: 'near', 'name', 'customer rating', 'area'


### EXAMPLE OF ALGORITHM IMPLEMENTATION ###

a) To re-train our best algorithm, the following script will work:
python learn_model --train_dataset <input_path> 
python test_model --test_dataset <input_path>

And the prediction will be available in the results folder.

b) To predict directly with our pre-trained model:

python test_model --test_dataset <input_path> --load saved_models/'64_100_40k.h5 --output_test_file results/pretrained_preds
But its predictions are already available in the results folder at 64_100_40k_results.


### FILES PRESENTATION ###

Our Solution for the 3rd NLP course assignment contains the following files and folders:
	1) learn_model.py
	2) test_model.py
	3) delexicalize.py
	4) data FOLDER
	5) saved_models FOLDER
	6) results
	7) temp

	1) learn_model.py:

The main file which contains our implementation of a seq2seq model.
It has the following arguments:
 --train_dataset: <input_path> for the training dataset
 --output_model_file: Save trained model weights to a '<output_path>.h5' file, default is 'saved_weights.h5'
The weights will be loaded afterwards by the test file
 --load: Load pre-trained model weights from the given path
 --model : choose the model seq2seq, either by character prediction "by_char" or by word prediction "by"_word". The default value is "by_char".

 
	2) test_model.py:

 --test_dataset <pathname_to_test_dataset> for test dataset
 --load <pathname_to_weights_file> Load pre-trained model weights from the given path. Load by default the last trained model from 'saved_weights.h5'
 --ouput_test_file <pathname_to_results_testfile> Save test predictions in csv or text file
 --model : choose the model seq2seq, either by character prediction "by_char" or by word prediction "by"_word". The default value is "by_char".


	3) delexicalize.py:

Personal implementation of a lexicalization function which enables to delexicalize and then relexicalize the input and target sentences to accelerate the learning.
e.g:

Original Meaning Representation: name[TheRiceBoat],food[Indian],priceRange [e20-25], customer rating [high], area [city centre], familyFriendly [yes], near [Express by Holiday Inn]

Original Natural LanguageSentences:  The Rice Boat is an Indian restaurant in the city centre near the Express by Holiday Inn, it is kid friendly highly rated and costs 20-25 euros.

De-lexicalized Meaning Representation: name [name x], food [Indian], priceRange [priceRange x], customer rating [customerRating x], area [city centre], familyFriendly [yes], near [near x]
De-lexicalized Natural Language Sentences: name x is an Indian restaurant in the city centre near near x, it is kid friendly customerRating x rated and costs priceRange x.

ref: TheE2ENLGChallenge: Training a Sequence-to-Sequence Approach for Meaning Representation to Natural Language Sentences, Elnaz Davoodi et al., Thomson Reuters.


	4) data FOLDER:

Which contains the train, test datasets


	5) saved_models FOLDER:

A folder containing already pre-trained weights, such as '64_100_40k.h5' which can be loaded with the --load argument in the test_model.py file. 
That is where the newly trained model weights will be saved as well.

	
	6) results FOLDER:

Folder which contains the predictions from the different models of the test dataset in csv.


	7) temp FOLDER:

Folder which contains a temporary file for loading the vocabulary from the learn_model.py file to the test_model.py csv.




