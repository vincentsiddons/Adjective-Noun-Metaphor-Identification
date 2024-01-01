import os
import numpy as np
import pandas as pd

class Preprocessing:
    csv = ''
    
    curr_directory = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, csv = os.path.join(curr_directory, 'AN-phrase-annotations.csv')):
        self.csv = csv
    
    @classmethod
    def no_arg(cls):
        return cls()
    
    #Creates array of inputs that is just each word
    def words_and_outputs(self):
        a_n_csv = open(self.csv, 'r')
        lines = a_n_csv.readlines()

        #get rid of \n at the end of each line and split each array line into csvs
        for i in range(0, len(lines)):
            lines[i] = lines[i][:len(lines[i]) - 1]
            lines[i] = lines[i].split(',')

        #create input array of each word and 0 or 1 outputs
        words = []
        for j in range(0, len(lines)):
            x = []
            for i in range(0, len(lines[j])):
                if lines[j][i].find("_") != - 1 and lines[j][i][lines[j][i].find("_") + 1:len(lines[j][i])] == "j":
                    x.append(lines[j][i][:len(lines[j][i]) - 1] + "ADJ")
                elif lines[j][i].find("_") != -1 and lines[j][i][lines[j][i].find("_") + 1:len(lines[j][i])] == "n":
                    x.append(lines[j][i][:len(lines[j][i]) - 1] + "NOUN")
                elif lines[j][i] == 'n':
                    x.append([0])
                elif lines[j][i] == 'y':
                    x.append([1])
            words.append(x)
        
        words = words[1:]
        words = sorted(words)

        return words
 
    #Creats column vector of concatenated Word2Vec embeddings from the English Gigaword Corpus
    def get_word2vec_embeddings(self):
        curr_directory = os.path.dirname(os.path.abspath(__file__))
        words = self.words_and_outputs()
        embeddings_thing = open(os.path.join(curr_directory, 'embeddings.txt'), 'r')
        
        try:
            embeddings_csv = open(os.path.join(curr_directory, 'new_embeddings.csv'), 'x')

            arr = []
            for lines in embeddings_thing:
                arr.append(lines)

            #In order for pandas to process the embeddings correctly, I have to add one comma deliminator between Words and Embeddings
            for i in range(0, len(arr)):
                arr[i] = arr[i].split()
                arr[i] = ",".join(arr[i][0:2]) + " " +  " ".join(arr[i][2:])
            arr[0] = "Word,Embedding"

            for i in range(0, len(arr)):
                embeddings_csv.write(arr[i] + "\n")

            embeddings_csv = open(os.path.join(curr_directory, 'new_embeddings.csv'), 'r')
        except:
            embeddings_csv = open(os.path.join(curr_directory, 'new_embeddings.csv'), 'r')

        embeddings_df = pd.read_csv(embeddings_csv, sep = ",", on_bad_lines = 'warn')
        
        #Create an array of concatinated embeddings for each adj-noun pair and add them to the words array
        for j in range(0, len(words) - 1):
            embedding = np.zeros(600)

            #get each embedding and replace string representations of floats with actual floats
            adj_emb = embeddings_df.loc[embeddings_df['Word'] == words[j][0], 'Embedding'].to_numpy()
            noun_emb = embeddings_df.loc[embeddings_df['Word'] == words[j][1], 'Embedding'].to_numpy()

            #np. fromstring is the best to use on space deliminated pandas series
            try:
                noun_emb = np.fromstring(noun_emb[0], sep= ' ')
                adj_emb = np.fromstring(adj_emb[0], sep=' ')
            except:
                continue
                
            #search dataframe for embedding and create a column vector of adjective and noun stacked on top of each other for processing
            embedding = np.concatenate((adj_emb, noun_emb))

            embedding = embedding[np.newaxis, :]
            embedding = embedding.transpose()
            words[j].append(embedding)

        embeddings_thing.close()
        embeddings_csv.close()
        
        return words
    