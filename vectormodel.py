#Jacob Hodgson

import math
import sys
import os
import string
import nltk

def main():

    doc_weight_scheme = 'tfidf'
    query_weight_scheme = 'tfidf'
    docs_to_index = "lepanto.txt"
    query_input = ''
    
    #handle arguments   
    if len(sys.argv) <= 1 or len(sys.argv) > 2:
        print("Need single argument to specify Tokenization. Please write \'true\' or \'false\'")
        exit(1)
    use_tokenizer = sys.argv[1]
    use_tokenizer = use_tokenizer.lower()
    if use_tokenizer != 'true' and use_tokenizer != 'false':
        print("Argument must be \'true\' or \'false\'")
        exit(1)

    if use_tokenizer == 'false':
        print("Warning: Not using a Tokenizer can potentially lead to a decrease in accuracy and query terms including punctuation.")
    
    #Variables
    token_list = []
    stop_token_list = []
    stem_token_list = []
    total_words = 0
    unique_set = set()
    unique_single_doc_set = set()
    inverted_index = {}
    user_input = ""

    doc_id = 0
    total_docs = 0
    total_unique = 0
    doc_vec_length = []
    temp_doc_length = 0

    #Create an Inverted Index, with O(1) Lookup Time. A line is considered a document
    with open((docs_to_index), encoding="utf8") as my_file:
        unique_single_doc_set = set()
        fileText = my_file.readlines()
        for line in fileText:
            recall_list.append(line)
            line = line.lower()
            inverted_index = indexDocument(line, inverted_index, doc_weight_scheme, query_weight_scheme, doc_id, use_tokenizer)
            doc_id += 1
            total_docs += 1

    #print(inverted_index)
    temp_counter = 0

    #Create a vector of document lengths for quicker calculations when taking user queries. 
    with open(((docs_to_index)), encoding="utf8") as my_file:
        fileText = my_file.readlines()
        for line in fileText:
            unique_single_doc_set = set()
            temp_doc_length = 0
            line = line.lower()
            if use_tokenizer == 'true':
                token_list = nltk.word_tokenize(line)
            else:
                token_list = line.split()
            #token_list = nltk.word_tokenize(line)
            stop_token_list = removeStopwords(token_list)
            stem_token_list = stemWords(stop_token_list)

            for token in stem_token_list:
                unique_single_doc_set.add(token)

            for token in unique_single_doc_set:
                #sqrt(sum of(tf*idf^2)
                #Euclidean Distance Calculation
                temp_doc_length += ((math.log(total_docs / inverted_index[token][0], 10) * inverted_index[token][1][temp_counter]) ** 2)
            temp_counter += 1  
            doc_vec_length.append(math.sqrt(temp_doc_length))

    total_unique = len(unique_set)
    
    while user_input != 'exit':
        user_input = input("Please enter a query. type \'exit\' to stop the program. ")
        user_input = user_input.lower()
        out_list = retrieveDocuments(user_input, inverted_index, doc_vec_length, doc_weight_scheme, query_weight_scheme, use_tokenizer)
        if not out_list and user_input != 'help' and user_input != 'exit':
            print("No matching documents, please try again. Type help for key insights")
        elif user_input == 'help':
            print("Stop words are over-valued in a tfidf vector-space model due to their frequency. Queries with only stop words are unmatchable.")
            print("Queries are formatted in a \"Bag of Words\" format meaning you can search any keywords you want in any order.")
            print("There is no case-matching. Hello and hello are treated the same both in documents and queries.")
            print("This program uses NLTK tokenization and PorterStemmer Stemming. \n https://github.com/peterwilliams97/nlp-class/blob/master/pa7-ir-v2/python/PorterStemmer.py \n https://www.nltk.org/api/nltk.tokenize.html")
            print("Valid arguments are \'true\' and \'false\'. If no Tokenizer selected, expect lower program accuracy.")
        for key, value in out_list.items():
            #print(str(line_counter+1) + ' ' + str(key+1) + ' ' + str(value) + '\n')
            print("Did you mean?: " + recall_list[key])
            break
            


            

def indexDocument(doc_string, inverted_index, doc_weight, query_weight, doc_id, use_tokenizer):
    token_list = []
    stop_token_list = []
    stem_token_list = []
    total_words = 0
    unique_set = set()
    unique_single_doc_set = set()
    final_stem_token_list = []
    idf_dict = {}
    tf_dict = {}

    #Inverted Index will look like {"Word": [Doc_Count, {Doc_id: Term Frequency in Document, Doc_id: Term Frequency in Document, ... , Final Doc_id: Term Frequency in Document}]}
    doc_string = doc_string.lower()
    if use_tokenizer == 'true':
        token_list = nltk.word_tokenize(doc_string)
    else:
        token_list = doc_string.split()

    stop_token_list = removeStopwords(token_list)
    stem_token_list = stemWords(stop_token_list)
    final_stem_token_list += stem_token_list

    for token in stem_token_list:
        total_words += 1
        unique_set.add(token)
        unique_single_doc_set.add(token)

    for token in unique_single_doc_set:
        if token not in inverted_index:
            inverted_index[token] = [1, {doc_id: stem_token_list.count(token)}]
        else:
            inverted_index[token][0] += 1
            inverted_index[token][1][doc_id] = stem_token_list.count(token)
    return inverted_index

def retrieveDocuments(query, inverted_index, doc_vec_length, doc_weight, query_weight, use_tokenizer):
    out_list = []

    if use_tokenizer == 'true':
        token_list = nltk.word_tokenize(query)
    else:
        token_list = query.split()
    stop_token_list = removeStopwords(token_list)
    stem_token_list = stemWords(stop_token_list)
    dot_product = 0
    query_doc_length = 0
    cos_sim = 0
    doc_set = set()
    token_set = set()
    ranking = {}
    query_list = {}
    token_weights = {}
    final_dict = {}

    for term in stem_token_list:
        if inverted_index.get(term, 0) != 0:
            #term is in index of words -->
            for each_doc in inverted_index[term][1]:
                doc_set.add(each_doc)
        token_set.add(term)
        if term not in query_list:
            query_list[term] = 1
        else:
            query_list[term] += 1

    #p(prob idf) = max{0, log (N-df_t)/df_t}

    #Calculate token weights of query in a dictionary
    for token in token_set:
        if inverted_index.get(token, 0) != 0:
            if(query_weight == 'tfidf'):
                #tf-idf of query token (TF * (IDF = N/df(document, term)))
                token_weights[token] = (query_list[token] * (math.log(150 / inverted_index[token][0], 10)))
        else:
            token_weights[token] = 0

    for each_doc in doc_set: #for each term in the query --> Sum of each weight for each doc
        cos_sim = 0
        for each_token in token_set:
            if inverted_index.get(each_token, 0) != 0:
                if inverted_index[each_token][1].get(each_doc, 0) != 0:
                    #Query Word TF-IDF * Document TF-IDF
                    cos_sim += token_weights[each_token] * ((math.log(150/ inverted_index[each_token][0], 10) * inverted_index[each_token][1][each_doc])) # tfidf Query Word * (tf * idf) Doc Word
            else:
                continue
        if doc_weight == 'tfidf':
            #Do final calculation for each document and rank them.
            ranking[each_doc] = round((cos_sim / (doc_vec_length[each_doc])), 5)

    out_list = sorted(ranking, key=ranking.__getitem__, reverse=True)
    for each_doc in out_list:
        final_dict[each_doc] = ranking[each_doc]
        #print(each_doc+1) 

    return (final_dict)


#Helper Function
def myFunc(e):
    return e['count']

#Helper Function
def remove_from_list(token_list, string):
    res = [i for i in token_list if i != string] 
  
    return res 

#Remove Stopwords
def removeStopwords(tokenlist):
    stop_list = []
    with open(("stopwords"), 'r') as my_file:
            fileText = my_file.read().replace("\n", " ")
    stop_list = fileText.split()
    for stopword in stop_list:
        tokenlist = remove_from_list(tokenlist, stopword)
    return tokenlist


#Stem our words using PorterStemmer
def stemWords(tokenlist):
    final_list = []
    stem = PorterStemmer()
    new_token = ""
    for token in tokenlist:
        new_token = stem.stem(token, 0, len(token)-1)
        final_list.append(new_token)

    return final_list
   


class PorterStemmer:

    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[k0],
        b[k0+1] ... ending at b[k]. In fact k0 = 0 in this demo program. k is
        readjusted downwards as the stemming progresses. Zero termination is
        not in fact used in the algorithm.

        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.
        """

        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0   # j is a general offset into the string

    def cons(self, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        """m() measures the number of consonant sequences between k0 and j.
        if c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        """vowelinstem() is TRUE <=> k0,...j contains a vowel"""
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        """doublec(j) is TRUE <=> j,(j-1) contain a double consonant."""
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j-1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        """cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. this is used when trying to
        restore an e at the end of a short  e.g.

           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        """ends(s) is TRUE <=> k0,...k ends with the string s."""
        length = len(s)
        if s[length - 1] != self.b[self.k]: # tiny speed-up
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k-length+1:self.k+1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        """setto(s) sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        """r(s) is used further down."""
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):   self.setto("ate")
            elif self.ends("bl"): self.setto("ble")
            elif self.ends("iz"): self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):   self.r("ate")
            elif self.ends("tional"):  self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):      self.r("ence")
            elif self.ends("anci"):    self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):       self.r("ble") # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):    self.r("al")
            elif self.ends("entli"):   self.r("ent")
            elif self.ends("eli"):     self.r("e")
            elif self.ends("ousli"):   self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):   self.r("ize")
            elif self.ends("ation"):   self.r("ate")
            elif self.ends("ator"):    self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):     self.r("al")
            elif self.ends("iveness"): self.r("ive")
            elif self.ends("fulness"): self.r("ful")
            elif self.ends("ousness"): self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):     self.r("al")
            elif self.ends("iviti"):   self.r("ive")
            elif self.ends("biliti"):  self.r("ble")
        elif self.b[self.k - 1] == 'g': # --DEPARTURE--
            if self.ends("logi"):      self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() dels with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):     self.r("ic")
            elif self.ends("ative"):   self.r("")
            elif self.ends("alize"):   self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):      self.r("ic")
            elif self.ends("ful"):     self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"): pass
            else: return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"): pass
            elif self.ends("ence"): pass
            else: return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"): pass
            else: return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"): pass
            else: return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"): pass
            elif self.ends("ible"): pass
            else: return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"): pass
            elif self.ends("ement"): pass
            elif self.ends("ment"): pass
            elif self.ends("ent"): pass
            else: return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'): pass
            elif self.ends("ou"): pass
            # takes care of -ous
            else: return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"): pass
            else: return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"): pass
            elif self.ends("iti"): pass
            else: return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"): pass
            else: return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"): pass
            else: return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"): pass
            else: return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k -1

    def stem(self, p, i, j):
        """In stem(p,i,j), p is a char pointer, and the string to be stemmed
        is from p[i] to p[j] inclusive. Typically i is zero and j is the
        offset to the last character of a string, (p[j+1] == '\0'). The
        stemmer adjusts the characters p[i] ... p[j] and returns the new
        end-point of the string, k. Stemming never increases word length, so
        i <= k <= j. To turn the stemmer into a module, declare 'stem' as
        extern, and delete the remainder of this file.
        """
        # copy the parameters into statics
        self.b = p
        self.k = j
        self.k0 = i
        if self.k <= self.k0 + 1:
            return self.b # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k+1]

if __name__ == "__main__":
    main()