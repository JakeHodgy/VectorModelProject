Jacob Hodgson - 3/26/2021 - Jhodg@Umich.edu
Interview Code Project Implementation

Task: Given a select document, allow users to input a query of words and suggest to them the correct line, or the closest line determined by our system.

Solution: This project uses a fairly standard vector-space model implementation. The Vector Space Model is an algebraic model for representing a collection of text documents as N-Dimensional Vectors in a Vector Space. The key insight here is we can also allow users to input specific queries and view these as vectors as well. Then using linear-algebra we can compare documents and queries.
One issue that comes up is document length. Not all documents are similar in length, and we allow users to specify their own queries, we must find a way to normalize document lengths into a specific value. We can calculate relevancy rankings of these documents to a user query using Document Cosine Similarity. Each document is given a score relating to each term (Word) that is pre-calculated in a quick-access container when we intially process our data set.

Key Equation: Cosine Similarity is calculated as (DotProduct of Two Vectors) / (Euclidean Distance between the Two Vectors to Normalize Length). 

The Weighting Scheme Chose is a TF-IDF weighting scheme for both document and query.
This is a very standard weighting scheme and the reasoning is Term Frequency is a very powerful tool; however, it assumes that all terms are equally important when assigning relevancy. When in fact many terms are not important in determing relevance (a term like "Adriatic" is MUCH more important than a term like "once" or "went"). So we scale down very common words (High Document Frequency) with Inverse Document Frequency, being log(N/df_term). 
TF*IDF.

Side Note: I considered using a different weighting scheme for my implementation as well. Term Frequency gives weighting to documents on a scale (1 time: 1x, 2 times: 2x, ..., 20 times: 20x); however, it is unlikely a term appearing twenty times is twenty times more valuable than a term appearing one time. So I looked at normalizng my Term Frequency by a factor of the Max Term Frequency of a document. I decided that because these lines are so short not only would normalization not a great change in results compared to an increased runtime, allowing user to specify their query length it would be more practical to take Term Frequency as a straight value. 
(For Example: User Query "Gun Gun" --> There are multiple queries with Gun, but only one query with Gun Gun so we can assume the user wants the "Gun Gun" document and give extra weighting to that query.)

To Run:
This program uses a PorterStemmer implementation for stemming words (https://github.com/peterwilliams97/nlp-class/blob/master/pa7-ir-v2/python/PorterStemmer.py). You do not need to install anything to run this as I have added the code myself in the file. The Porter Stemmer Class is not my code.

This program also uses a NLTK tokenizer implementation. The documentation can be found here (https://www.nltk.org/api/nltk.tokenize.html)
The installation instructions for NLTK Python can be found here (https://www.nltk.org/install.html)
You also need to install NLTK data locally for the tokenizer to work. (https://www.nltk.org/data.html)

I believe the rest of the packages are standard for Python3.
I also have included a small, standard stopwords file to help remove stopwords. You need this file to run the project.

My local environment is on Windows, so for my installation I used Python 3.8.8, and installed packages using PIP package-manager. Similarily for a linux version you can just Git Clone all the files, install your Python packages, and run. I also tested it on a Linux distribution and it worked as well.

Running the program with NLTK Package: python3 vectormodel.py true

Running the program without the NLTK Package: python3 vectormodel.py false

Extra: Given that some users may not be able to install the NLTK Package or its Data whether that be space requirement or security requirements. I have added an extra argument that allows users to run the program with Tokenization. Given extra time, I would be able to implement my own tokenizer; however, it is a time consuming task to handle every single tokenization case properly. You will see potential accuracy differences without a Tokenizer due to punctuation.


