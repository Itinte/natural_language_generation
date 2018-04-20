from __future__ import print_function
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import sys
import csv
import re
import os
from delexicalize import delex, relex
import argparse
import random


parser = argparse.ArgumentParser(description='seq2seq Testing Script')
parser.add_argument('--test_dataset', default='data/testset.csv',type = str, help='<input_path> for the test dataset')
parser.add_argument('--output_test_file', default='results/testset_result.csv', type = str, help='Save trained model result of testset file')
parser.add_argument('--load', type =str, default= 'saved_models/saved_weights.h5', help='Load pre-trained model weights from the given path')
parser.add_argument('--model', type=str, help="Two models possible : by_char or by_word", default="by_char")

class seq2seq():

    def __init__(self,model="by_char"):

        # MODEL HYPERPARAMETERS
        self.latent_dim = 100  # Latent dimensionality of the encoding space..

        #TYPE OF MODEL
        self.model=model

        # CLASS DATA STORING
        self.input_texts_src = [] # List of the original input sentences as read from the dataset and not modified
        self.target_texts_src =[] # List of the original target sentences as read from the dataset and not modified
        self.input_texts = [] # List of the input sentences AFTER DELEXICALIZATION
        self.target_texts = [] # List of the target sentences AFTER DELEXICALIZATION
        self.input_dic = [] # List of the input sentences dictionaries storing the original MR, necessary for relexicalization
        #by character
        self.input_characters = set() # List of the differents characters present in the input sentences
        self.target_characters = set() # List of the differents characters present in the target sentences
        # by word
        self.input_words = set()
        self.target_words = set()

        # LEXICALIZATION CATEGORIES 
        self.slots = ['near', 'name', 'customer rating', 'area'] # List of the slots used for the delexicalisation
        #["name", "near", "area", "eatType", "priceRange", "familyFriendly", "food", "customer rating"]


    def preprocess(self, path): # Preprocess dataset: store sentences, delexicalization, characters parsing, character indexes creation 

        
        # Load Dataset
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f,delimiter=',') # Open file and define a reader to parse the lines
            next(reader) #ignore head
        # parse dataset line by line
            if self.model=="by_char":

                for input_text in reader:
                    
                    self.input_texts_src.append(input_text) # Store the original input sentence
                    target_text="" # Store the original target sentence
                    sent_dict, input_text, target_text = delex(input_text[0], target_text, self.slots) # Delexicalize and compute the sentences dictionaries
                    self.input_texts.append(input_text) # Store the delexicalized input sentence
                    self.input_dic.append(sent_dict) # Store the input dictionaries to allow relexicalization

                    # Get the characters list for input/target
                    for char in input_text: # Parse delexicalized input sentence char by char
                        if char not in self.input_characters:
                            self.input_characters.add(char) # Add it to the list
                    
            else:

                for input_text in reader:
                    self.input_texts_src.append(input_text)
                    target_text=""
                    sent_dict, input_text, target_text = delex(input_text[0], target_text, self.slots)
                    input_text=re.sub(' +',' ',input_text)
                    input_text = re.sub(r'[^\w\s]','',input_text)
                    input_text=input_text.split(" ")
                    self.input_texts.append(input_text)
                    self.input_dic.append(sent_dict)

                    for word in input_text:
                        if word not in self.input_characters:
                            self.input_characters.add(word)
     
        
        # Sort characters lists for readability
        self.input_characters = sorted(list(self.input_characters))
        character_list=np.load("temp/temp_file.npy")
        
        #os.remove('temp_file.npy')
        self.target_characters=character_list#.to_list()
        
        # Define important variables
        self.num_encoder_tokens = len(self.input_characters) # Number of different characters in the list of input sentences
        self.num_decoder_tokens = len(self.target_characters) # Number of different characters in the list of target sentences
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts]) # Maximum length of an input sequence (char size)
        if self.model=="by_char":
            self.max_decoder_seq_length = 300
        else:
            self.max_decoder_seq_length=295
        print()
        print('DATASET VARIABLES')
        print('Number of samples:', len(self.input_texts))
        
        
        # Create CHAR index for input and target sentences lists
        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])
        
        # Reverse-lookup CHAR index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

        # Close Dataset
        f.close()


    def create_dataset(self,batch_size): # Create the necessary dataset for Encoder/Decoder

        if self.model=="by_char":
            # Create the encoder input, decoder input and decoder target data necessary for the learning
            encoder_input_data = np.zeros(
                (len(self.input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
                dtype='float32')
            

            for i, input_text in enumerate(self.input_texts):
                for t, char in enumerate(input_text):
                    encoder_input_data[i, t, self.input_token_index[char]] = 1.
        else:

            encoder_input_data=np.zeros((self.num_samples,self.max_encoder_seq_length,self.num_encoder_tokens),dtype="int32")
            
            #batch_id = random.sample(range(len(self.input_texts)),batch_size)
           
            for j,i in enumerate(self.num_samples):
            
                for index, word in enumerate(self.input_texts[i]):
                    encoder_input_data[j,index,self.input_token_index[word]]=1
                    
               
        return encoder_input_data
    
    

    def load_weights(self, weights_path): # Load a pre-trained model from input path

        print()
        print('LOADING MODEL...')
        # Restore the model and construct the encoder and decoder.
        model = load_model(weights_path)

        encoder_inputs = model.input[0]   # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]   # input_2
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model


    def decode_sequence(self, input_seq, encoder_model, decoder_model): # Decode a single input sentence

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        if self.model=="by_char":
            target_seq[0, 0, self.target_token_index['\t']] = 1.
        else:
            target_seq[0, 0, self.target_token_index['start_']] = 1
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            
            if self.model=="by_char":

                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length):
                    stop_condition = True
               
            else:

                decoded_sentence += sampled_char+" "

                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == 'end_' or len(decoded_sentence) > self.max_decoder_seq_length):
                    stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


    def apply_seq2seq(self, encoder_input_data, encoder_model, decoder_model, path_output): # Apply trained seq2seq on 'n_range' indexes of the dataset:

        print()
        print('STARTING PREDICTING FOR' + str(len(self.input_texts_src))+' INPUT SENTENCES...')
        print()
        with open(path_output, 'w', encoding='utf-8') as f:
        
            for seq_index in range(len(self.input_texts_src)):
                print('Sentence n-', seq_index)
               
                input_seq = encoder_input_data[seq_index: seq_index + 1]  # Take one sequence (part of the training set)
                decoded_sentence = self.decode_sequence(input_seq, encoder_model, decoder_model) # Decode input sequence
                decoded_sentence = relex(self.input_dic[seq_index], decoded_sentence) # Relexicalisation of the decoded sentence:
                f.write(decoded_sentence+"\n")
                print('---')
                print('Input sentence:', self.input_texts_src[seq_index])
                print('Decoded sentence:', decoded_sentence)
                
        f.close()


def main(): # Main function of the file acting which run the training/loading

    args = parser.parse_args() # Parse script arguments
    seq = seq2seq(args.model) # Instantiate model
    seq.preprocess(args.test_dataset) # Preprocess Dataset (necessary even when only loading pre-trained model)
    encoder_input_data = seq.create_dataset(500) # Create test data

    # Load pre-trained weights
    encoder_model, decoder_model = seq.load_weights(args.load) # 'PATH EXAMPLE: saved_models/80_100_20k.h5'

    seq.apply_seq2seq(encoder_input_data, encoder_model, decoder_model, args.output_test_file) # Apply trained model on a chunk sentences


if __name__ == "__main__":
    main()