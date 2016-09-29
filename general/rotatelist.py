#!/bin/python3
#hackerrank algorithm warmup
import sys

#first input line
print("Enter n, k, q followed by spaces")
print('n = # integers, k = # rotations, q = # queries')
n,k,q = input().strip().split(' ')
n,k,q = [int(n),int(k),int(q)]
#n integers
#k right rotations
#q queries

#second input line
a = [int(a_temp) for a_temp in input().strip().split(' ')]
#rotate list by k to the right
def rotate_list(mylist, k):
    kreduced = k%len(mylist)
    return mylist[-kreduced:]+mylist[:-kreduced]

b = rotate_list(a, k)
#print value of b_m
for a0 in range(q):
    m = int(input().strip())
    print(b[m])