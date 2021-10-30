# text_mining-
python code that filter tweets 

text_mining:
This code performs the preprocessing of the data mining. The code starts by
finding all the possible classes, the second most common class and which date 
has the most positive sentiment. Then it filters the text by removing 
non-alphabetical characters, setting all letters to lower case and 
removing unnecessary spaces. Then the words are tokenized and filtered by 
removing stop words and all words with 2 or fewer characters. The 
difference in the number of words and most common words are shown before and
after filtering. Then a histogram is made showing the word frequency across the 
documents. The data is used to form a term-document matrix to train a
Multinomial Naive Bayes classifier to classify text. 
