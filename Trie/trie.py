'''
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

    Trie() Initializes the trie object.
    void insert(String word) Inserts the string word into the trie.
    boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
    boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

'''
class Trie:

    def __init__(self):
        self.root = {}
        

    def insert(self, word: str) -> None:
        trie = self.root
        for c in word:
            if c in trie:
                trie = trie[c]
            else:
                trie[c] = {}
                trie = trie[c]
        trie['end'] = True
        

    def search(self, word: str) -> bool:
        trie = self.root
        for c in word:
            if c in trie:
                trie = trie[c]
            else:
                return False
        if 'end' in trie:
            return True
        return False
        

    def startsWith(self, prefix: str) -> bool:
        trie = self.root
        for c in prefix:
            if c in trie:
                trie = trie[c]
            else:
                return False
        return True
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)



'''
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

    WordDictionary() Initializes the object.
    void addWord(word) Adds word to the data structure, it can be matched later.
    bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.


'''

# Can reimplemnet such that def init is a dict and isWord and we recursively add WordDictionary objects
class WordDictionary:

    def __init__(self):
        self.trie = {}
        

    def addWord(self, word: str) -> None:
        curr_trie = self.trie
        for c in word:
            if c in curr_trie:
                curr_trie = curr_trie[c]
            else:
                curr_trie[c] = {}
                curr_trie = curr_trie[c]
        curr_trie['isWord'] = True

    def search(self, word: str, trie=None) -> bool:
        if trie == None:
            curr_trie = self.trie
        else:
            curr_trie = trie
        for i, c in enumerate(word):
            if c == '.':
                for ch in curr_trie.keys():
                    if ch == 'isWord': continue
                    if self.search(word[i+1:], curr_trie[ch]):
                        return True
                return False
            if c not in curr_trie:
                return False
            curr_trie = curr_trie[c]
        
        return 'isWord' in curr_trie
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)