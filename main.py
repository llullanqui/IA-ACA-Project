import keras
import numpy
import random
import tkinter
from tkinter import ttk
from functools import partial

datasetFilePath = "updatedDataset.csv"
modelFilePath = "edificationEvaluationModel.hdf5"
currentResult = ""

#Dictionaries to give a number to the data in 'UpdatedDataset.csv'
edificationType = {
	"tipconst_hormigo":1,
	"tipconst_mixta":2,
	"tipconst_estmad":3,
	"tipconst_estmuro":4,
	"tipconst_manrest":5,
	"tipconst_metalic":6,
	"tipconst_modpre":7,
	"tipconst_refestruc":8
}

yesOrNo = {
	"si":0,
	"no":1
}

qualification = {
	"poca":0,
	"mode":1,
	"sev":2
}

result = {
	"USO RESTRINGIDO":1,
	"INSEGURO":0,
	"INSPECCIONADO":2
}

#Reads the data 
def loadDatasetEqualParts(filename):
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	with open(filename) as file:
		counter0 = 0
		counter1 = 0
		counter2 = 0
		file.readline()
		for line in file:
			line = line.strip("\n").split(";")
			vectorX = [edificationType[line[0]],
			yesOrNo[line[1]],
			yesOrNo[line[2]],
			yesOrNo[line[3]],
			qualification[line[4]],
			qualification[line[5]],
			qualification[line[6]],
			qualification[line[7]],
			qualification[line[8]]]
			flag = False
			if result[line[9]]==0 and counter0 < 14000:
				counter0 += 1
				flag = True
			if result[line[9]]==1 and counter1 < 14000:
				counter1 += 1
				flag = True
			if result[line[9]]==2 and counter2 < 14000:
				counter2 += 1
				flag = True
			vectorY = [0]*3
			vectorY[result[line[9]]] = 1
			if flag:
				x_train.append(vectorX)
				y_train.append(vectorY)
			else:
				x_test.append(vectorX)
				y_test.append(vectorY)
	
	return (numpy.array(x_train,dtype=float),numpy.array(y_train,dtype=float)),(numpy.array(x_test,dtype=float),numpy.array(y_test,dtype=float))

def loadDatasetNotEqualParts(filename):
	x = []
	y = []
	with open(filename) as file:
		file.readline()
		for line in file:
			line = line.strip("\n").split(";")
			vectorX = [edificationType[line[0]],
			yesOrNo[line[1]],
			yesOrNo[line[2]],
			yesOrNo[line[3]],
			qualification[line[4]],
			qualification[line[5]],
			qualification[line[6]],
			qualification[line[7]],
			qualification[line[8]]]
			x.append(vectorX)
			y.append(result[line[9]])
	x = numpy.array(x)
	y = numpy.array(y)
	y = keras.utils.to_categorical(y)
	trainAmount = int(len(x)/100)*90
	x_train = x[:trainAmount]
	x_test = x[trainAmount:]
	y_train = y[:trainAmount]
	y_test = y[trainAmount:]
	
	return (x_train,y_train),(x_test,y_test)

def trainModel():
	#We load the data
	(x_train,y_train),(x_test,y_test) = loadDatasetEqualParts(datasetFilePath)
	###(x_train,y_train),(x_test,y_test) = loadDatasetNotEqualParts(datasetFilePath)

	# Model building
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(81, activation='relu',input_dim=9,bias_initializer='ones'))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(81, activation='relu'))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(3, activation='softmax'))

	#model.summary() #Commented to avoid filling up the cmd screen

	# Model compilation
	model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

	# We evaluate the test data in the model. We use the first 75% rows of the test data.
	score = model.evaluate(x_test[:int(len(x_test)*.75)],y_test[:int(len(x_test)*.75)], batch_size=128)
	accuracy = 100*score[1]
	print('Precisión del modelo: %.4f%%' % accuracy)

	checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelFilePath, verbose=1, save_best_only=True)
	# Model training
	model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1, callbacks=[checkpointer], verbose=1, shuffle=True)

	model.load_weights(modelFilePath)

	# We evaluate the test data in the trained model. We use the first 75% rows of the test data.
	score = model.evaluate(x_test[:int(len(x_test)*.75)],y_test[:int(len(x_test)*.75)], batch_size=128)
	accuracy = 100*score[1]
	print('Precisión del modelo: %.4f%%' % accuracy)

def dataInputWindow():
	window = tkinter.Tk()
	window.title("Evaluar edificación")

	variableMaterial = tkinter.StringVar()
	labelMaterial = tkinter.Label(window, text="Tipo de material: ")
	labelMaterial.grid(column=0,row=0)
	comboMaterial = ttk.Combobox(window, textvariable=variableMaterial, values = (
		"Hormigón",
		"Mixta",
		"Madera",
		"Muro",
		"Restos",
		"Metalica",
		"Modpre",
		"Estructural"))
	comboMaterial.current(0)
	comboMaterial.grid(column=1,row=0)

	labelColapsoTotal = tkinter.Label(window, text="Colapso total: ")
	labelColapsoTotal.grid(column=0,row=1)
	comboColapsoTotal = ttk.Combobox(window,values = [
		"No",
		"Sí"])
	comboColapsoTotal.current(0)
	comboColapsoTotal.grid(column=1,row=1)

	labelColapsoParcial = tkinter.Label(window, text="Colapso parcial: ")
	labelColapsoParcial.grid(column=0,row=2)
	comboColapsoParcial = ttk.Combobox(window,values = [
		"No",
		"Sí"])
	comboColapsoParcial.current(0)
	comboColapsoParcial.grid(column=1,row=2)

	labelCimentacionAfectada = tkinter.Label(window, text="¿Cimentación afectada?: ")
	labelCimentacionAfectada.grid(column=0,row=3)
	comboCimentacionAfectada = ttk.Combobox(window,values = [
		"No",
		"Sí"])
	comboCimentacionAfectada.current(0)
	comboCimentacionAfectada.grid(column=1,row=3)

	labelParam1 = tkinter.Label(window, text="FUERA_PLOM: ")
	labelParam1.grid(column=0,row=4)
	comboParam1 = ttk.Combobox(window,values = [
		"Poco",
		"Moderado",
		"Severo"])
	comboParam1.current(0)
	comboParam1.grid(column=1,row=4)

	labelParam2 = tkinter.Label(window, text="AGR_MUROS: ")
	labelParam2.grid(column=0,row=5)
	comboParam2 = ttk.Combobox(window,values = [
		"Poco",
		"Moderado",
		"Severo"])
	comboParam2.current(0)
	comboParam2.grid(column=1,row=5)

	labelParam3 = tkinter.Label(window, text="AGR_MAMP: ")
	labelParam3.grid(column=0,row=6)
	comboParam3 = ttk.Combobox(window,values = [
		"Poco",
		"Moderado",
		"Severo"])
	comboParam3.current(0)
	comboParam3.grid(column=1,row=6)

	labelParam4 = tkinter.Label(window, text="DASOBCH: ")
	labelParam4.grid(column=0,row=7)
	comboParam4 = ttk.Combobox(window,values = [
		"Poco",
		"Moderado",
		"Severo"])
	comboParam4.current(0)
	comboParam4.grid(column=1,row=7)

	labelParam5 = tkinter.Label(window, text="OTROS: ")
	labelParam5.grid(column=0,row=8)
	comboParam5 = ttk.Combobox(window,values = [
		"Poco",
		"Moderado",
		"Severo"])
	comboParam5.current(0)
	comboParam5.grid(column=1,row=8)

	calculate = tkinter.Button(window, text="Calcular estado",command=partial(predictResult,
		comboMaterial.current(),
		comboColapsoTotal.current(),
		comboColapsoParcial.current(),
		comboCimentacionAfectada.current(),
		comboParam1.current(),
		comboParam2.current(),
		comboParam3.current(),
		comboParam4.current(),
		comboParam5.current()))
	calculate.grid(column=0,row=9)

	window.mainloop()

def predictResult(x0,x1,x2,x3,x4,x5,x6,x7,x8):
	print(x0,x1,x2,x3,x4,x5,x6,x7,x8)


print("\nEvaluación de vivienda/edificación después de catástrofes.")
menu = """\nEscoja una opción:
1. Entrenar red neuronal
2. Evaluar edificación
3. Salir"""

option = ""
while option != "3":
	print(menu)
	option = input("\nEscriba el número de la opción a escoger: ")
	if option == "1":
		trainModel()
	elif option == "2":
		dataInputWindow()
	elif option == "3":
		print("\n¡Por un Ecuador amazónico!\n¡Desde siempre, y hasta siempre!\n¡VIVA LA PATRIA!\n")
	else:
		print("Ingrese una opción correcta: 1, 2, o 3.")





"""
result=model.predict(x_test[int(len(x_test)*.75):])
labels = (result > 0.5).astype(numpy.int)
print(labels)
"""