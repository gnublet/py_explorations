#converts standard time to military time
#input: hh:mm:ss_M where 0<=hh<=12
#output: hh:mm:ss where 0<=hh<24
import sys

time = input().strip()
mystrs = time.split(sep=':')
aorp = time[-2]
if aorp == 'A':
    hh = int(mystrs[0])%12
    print(str(hh).rjust(2,'0') + time[2:8])
elif aorp == 'P':
    hh = int(mystrs[0])
    if hh==12:
        print(str(hh).rjust(2,'0') + time[2:8])
    else:
        print(str(   (int(mystrs[0])+12)  %24).rjust(2, '0')+":"+mystrs[1] + ":"+mystrs[2][0:2])
else:
	print('Not correct time format')