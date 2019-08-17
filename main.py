#import keras
import numpy
import random

dictProvs = dict()
dictCities = dict()
filename = "SouthAmerica.csv"

#Method to generate numbers for Provinces and Cities.
#The initial dataset didn't have them.
def generateIdsForProvsAndCities():
	counterProvs = 0
	counterCities = 0
	with open(filename) as file:
		file.readline()
		for line in file:
			line = line.strip("\n").split(";")
			if dictProvs.get(line[11],0)==0:
				counterProvs += 1
				dictProvs[line[11]] = counterProvs
			if dictCities.get(line[12],0)==0:
				counterCities += 1
				dictCities[line[12]] = counterCities

#Method to generate a dataset with just the data we'll use to train the model
def writeProvsAndCitiesNumbers():
	generateIdsForProvsAndCities()
	initialDataset = open(filename)
	updatedDataset = open("UpdatedDataset.csv","w+")
	line = initialDataset.readline().strip("\n").split(";")
	line = ";".join([line[2],line[3],line[7],line[8],"prov",line[11]+"_txt","city",line[12]+"_txt",line[13],line[14],line[28],line[29]+"\n"])
	updatedDataset.write(line)
	for line in initialDataset:
		line = line.strip("\n").split(";")
		line = ";".join([line[2], line[3], line[7], line[8], str(dictProvs[line[11]]), line[11], str(dictCities[line[12]]), line[12], line[13], line[14], line[28],line[29]+"\n"])
		updatedDataset.write(line)
	updatedDataset.close()
	initialDataset.close()

def loadDataset(filename):
	x = []
	y = []
	with open(filename) as file:
		file.readline()
		for line in file:
			line = line.strip("\n").split(";")
			if(line[9]==""):
				line[9] = str(round(random.uniform(-90,90),6))
			if(line[8]==""):
				line[8] = str(round(random.uniform(-90,90),6))
			vectorX = [int(line[0]),int(line[1]),int(line[2]),int(line[4]),int(line[6]),float(line[8]),float(line[9])]
			vectorY = numpy.zeros(9)
			vectorY[int(line[10])-1] = 1
			x.append(vectorX)
			y.append(vectorY)
	x = numpy.array(x)
	y = numpy.array(y)
	trainAmount = int(len(x)/10)*9
	x_train = x[:trainAmount]
	x_test = x[trainAmount:]
	y_train = y[:trainAmount]
	y_test = y[trainAmount:]
	return (x_train,y_train),(x_test,y_test)

(x_train,y_train),(x_test,y_test) = loadDataset("updatedDataset.csv")
