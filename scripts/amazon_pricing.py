#!/usr/bin/python3

"""
looks up pricing of aws instances
"""
import pandas as pd

class Pricing(object):

    def __init__(self):
        self.lookup_table = pd.DataFrame.from_csv('../data/pricing.csv')
        self.current_table = self.lookup_table.copy()

    def find(self, query):
       self.current_table = self.lookup_table.copy()
       for key, value in query.items():
           try:
               # for each criterion filter the current dataframe with it
               self.current_table = self.current_table[self.current_table[key] == value]
           except KeyError:
               print("queries must be one of the following: ['vcpu', 'ecu', 'memory', 'ssd', 'storage', 'num_drives', 'price','networking', 'gpu']")
               return 0

       try:
           # return only the cheapest price
           return min(list(self.current_table['price']))
       except ValueError:
           # empty list / not defined
           return 0


#run tests
if __name__ == "__main__":
    p = Pricing()
    print('*************test 1******************')
    print('query with a bunch of matches')
    print('*************************************')
    q = {'gpu' : 0}
    print(p.find(q))
    print()
    print('*************test 2******************')
    print('query with a single match')
    print('*************************************')
    q = {'gpu' : 1, 'networking': 'max'}
    print(p.find(q))
    print()
    print('*************test 3******************')
    print('query with no matches')
    print('*************************************')
    q = {'vcpu': 2, 'gpu' : 1, 'networking': 'max'}
    print(p.find(q))
    print()
    print('*************test 4******************')
    print('query with error')
    print('*************************************')
    q = {'gpupu' : 1, 'networking': 'max'}
    print(p.find(q))
