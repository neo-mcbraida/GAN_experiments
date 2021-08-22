import csv

g = [[0, 9, 6, 4], [9, 6], [8, 5, 8, 3, 2, 7]]

with open('C:/Users/nsmne/Documents/GAN_experiments/inputData.csv', 'w', newline='') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
   # write.writerow(fields)
    write.writerows(g)