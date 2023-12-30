This program impliments the neural network found in Section 3.2 of the article "“Deep” Learning: Detecting Metaphoricity in Adjective-Noun Pairs" by Bizzoni et al. 
The details concerning it's composition can be found in the paper. My model calculates the f-score given three hyperperameters: learning rate, epochs, and the leaky ReLU negative slope.

I used pretrained adjective, noun embeddings from the English Gigaword corpus that can be found here: http://vectors.nlpl.eu/explore/embeddings/en/models/
Adjectives all had embeddings corresponding to themselves, but not all nouns did. I just skipped those missing in training, validation, and testing. 

I did not include this file inside my repository. For the program to run you MUST download it. After downloading it:

1. Drag it into the same folder as the rest of the files.
2. Rename the file "embeddings."(do not include the period and the quotation marks in the name) Make sure it's in the txt file format. 
