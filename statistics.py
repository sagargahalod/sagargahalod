import statistics as s
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mydata = [1,2,3,5,15,30,35,40,50,100,250,2,2,3,2,2,2]

print(s.mean(mydata))
print(s.median(mydata))
print(s.mode(mydata))