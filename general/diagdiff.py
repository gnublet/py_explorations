#!/bin/python3

import sys


n = int(input().strip())
a = []
for a_i in range(n):
    a_t = [int(a_temp) for a_temp in input().strip().split(' ')]
    a.append(a_t)

primarydiag = 0
secondarydiag = 0
for i in range(n):
	primarydiag += a[i][i]
	secondarydiag += a[-i][i]
