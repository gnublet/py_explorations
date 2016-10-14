#import the argv method of the sys library
from sys import argv

#first parameter is argv[0] which contains the script name
#second parameter is argv[1] which contains the file name
script, filename = argv
#just printing for instructional purposes
#print(script,filename)
#print(argv)
#print(argv[0],argv[1])

#open the file
txt = open(filename)

#print the filename
print("Here's your file {}".format(filename))
#read and print what's in the filename file
print(txt.read())

#ask user for the same file (could be another file)
print("Type another: ")
file_again = input("> ")
#read the contents
txt_again = open(file_again)
#print what is read from the file
print(txt_again.read())

#close the files
txt.close()
txt_again.close()