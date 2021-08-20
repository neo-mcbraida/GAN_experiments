import re
from typing import Dict

def load_words():
    with open('words_alpha.txt') as word_file:
        valid_words = list(word_file.read().split())

    return valid_words


#if __name__ == '__main__':
  #  english_words = load_words()
    # demo print
   # print('fate' in english_words)

words = load_words()
dictWords = {i:words[i] for i in range(1, len(words))}
dictWords[0] = ""
print(dictWords[1])

#for i in range(1, len(words)):#
    #dictWords.update( {i : words[i]})
    
#print(dictWords[0], dictWords[1])


"""
with open('movie_lines.txt') as f:
    lines = f.readlines()

temp = []
for line in lines:
    line = line[36:]
    i = 0
    u = 0
    for char in line:
        i += 1
        if char == "+":
            u = i
    line = line[u:]
    line = re.sub(r"[']", '', line)
    line = re.sub(r"[^a-zA-Z]+", ' ', line)
    temp.append(line.split())
lines = temp

print(lines)
"""