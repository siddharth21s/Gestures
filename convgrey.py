import os
from PIL import Image

def convert2grey(name):
	img = Image.open(name).convert('L')
	img.save(name)
	

def main():
	currpath = os.getcwd()
	print(currpath)
	input("Proceed1")
	alldir = [n for n in os.listdir(".") if os.path.isdir(n)]
	print(alldir)
	input("Proceed2")
	'''
	for i in alldir:
		os.chdir(currpath+"/"+i)
		print(os.getcwd())
		input("Proceed3")
		'''
	os.chdir(currpath+"/"+alldir[1])
	allfiles = [n for n in os.listdir(".") if os.path.isfile(n)]
	print(allfiles)
	input("Proceed4")	
	for j in allfiles:
		convert2grey(j)
		
	'''	
	allfiles = [n for n in os.listdir(".") if os.path.isfile(n)]
	print(allfiles)
	input("Proceed")
	for i in allfiles:
		convert2grey(i)
	'''

if __name__ == "__main__":
	main()
