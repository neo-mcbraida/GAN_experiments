import re
import csv
import os

def load_words():
    valid_words = [""]
    with open('words_alpha.txt') as word_file:
        valid_words += list(word_file.read().split())

    return valid_words


words = load_words()#{i:words[i] for i in range(1, len(words))}

diction = {i:words[i] for i in range(1, len(words))}

#for i in range(len(diction)):
 # words.append(diction[i])

with open('movie_lines.txt') as f:
    lines = f.readlines()

temp = []

def getLineNum(line):
    line = line[1:]
    num = ""
    for char in line:
        if char == " ":
            break
        else:
            num += char
    return int(num)        

convo = []
prevLineNum = 0

for line in lines:
    #line = line[36:]
    i = 0
    u = 0
    lNum = getLineNum(line)
    if (lNum + 1) != prevLineNum:
        temp.insert(0, convo)
        convo = []
    prevLineNum = lNum
    for char in line:
        i += 1
        if char == "+":
            u = i
    line = line[u:]
    line = re.sub(r"[']", '', line)
    line = re.sub(r"[^a-zA-Z]+", ' ', line)
    line = line.lower()
    line = line.split()
    l = [words.index(word) for word in line if word in words]
    convo.append(l)
lines = temp

print(lines[0])
#words.index to get key
with open('inputData.csv', 'w', newline='') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
   # write.writerow(fields)
    write.writerows(lines)