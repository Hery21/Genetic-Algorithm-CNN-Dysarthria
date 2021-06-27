import numpy as np
import random
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import rmsprop

def fitnesscnn(individual, X_train, Y_train, x_test, y_test):
  num_hidden = 3

  if int(individual[1][0]) == 0:
    num_hidden-=1

  if int(individual[2][0]) == 0:
    num_hidden-=1

  model = Sequential([])

  for j in range(num_hidden):
    if int(individual[j][0]) == 0:
      j+=1
    model.add(Conv2D(int(individual[j][1]), int(individual[j][0]), padding='same', input_shape=(186, 13, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
  
  model.add(Flatten())
  model.add(Dense(10, activation='softmax'))
  
  opt = rmsprop(learning_rate=0.001, decay=1e-6)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=opt,
      metrics=['accuracy'],
      )
  history = model.fit(
    X_train,
    to_categorical(Y_train),
    verbose=0,
    epochs=20,
    batch_size=32,
    validation_data=(x_test, to_categorical(y_test)),
    )

  fitness = np.array(history.history['val_accuracy'][-1])
  model.save('model/percobaan 2/M14/' + str(individual) + str(fitness*100) + '.h5')
  del model
  return fitness

def population_init(num_pop):
  limit = {0: [0,3,5,7],
           1: [5,10,12,15,20]
           }

  population = []

  for i in range(num_pop):
    init_crom = [[[np.random.choice(limit[0][1:]), np.random.choice(limit[1])],
                  [np.random.choice(limit[0]), np.random.choice(limit[1])],
                  [np.random.choice(limit[0]), np.random.choice(limit[1])]
                  ], i]
    if i>0:
      for j in range(i):
        while init_crom == population[j][0]:
          init_crom = [[[np.random.choice(limit[0][1:]), np.random.choice(limit[1])],
                        [np.random.choice(limit[0]), np.random.choice(limit[1])],
                        [np.random.choice(limit[0]), np.random.choice(limit[1])]
                        ], i]
        else:
          continue
      else:
        population.append(init_crom)
    else:
      population.append(init_crom)

  return np.array(population)

def cross(population):
  #memilih 2 individu secara random dari populasi
  index1 = np.random.choice(len(population), 2, replace=False)

  #mengcopy kromosom agar tidak terjadi kesalahan by reference
  cromosome1 = np.copy(population[index1[0]][0])
  cromosome2 = np.copy(population[index1[1]][0])

  #memilih point crossover secara random
  index2 = random.randint(1, 2)

  #crossover
  cross1 = np.concatenate((cromosome1[0:index2], cromosome2[index2:len(cromosome2)]))
  cross2 = np.concatenate((cromosome2[0:index2], cromosome1[index2:len(cromosome1)]))

  return cross1, cross2

def mutate(cross1, cross2, pm):
  limit = {0: [0,3,5,7],
           1: [5,10,12,15,20]
           }
  
  #memilih titik-titik mutasi
  indexlist = []
  for j in range(len(cross1)):
    for k in range(len(cross1[0])):
      indexlist.append([j,k])

  index3 = np.random.choice(len(indexlist), int(pm * len(indexlist)), replace = False)
  index4 = np.random.choice(len(indexlist), int(pm * len(indexlist)), replace = False)
  
  #mutasi
  for l in range(len(index3)):
    if indexlist[index3[l]][0] == 0 and indexlist[index3[l]][1] == 0:
      new_gene1 = random.choice(limit[indexlist[index3[l]][1]][1:])
      while new_gene1 == cross1[indexlist[index3[l]][0]][indexlist[index3[l]][1]]:
        new_gene1 = random.choice(limit[indexlist[index3[l]][1]][1:])
      else:
        cross1[indexlist[index3[l]][0]][indexlist[index3[l]][1]] = new_gene1
    else:
      new_gene1 = random.choice(limit[indexlist[index3[l]][1]])
      while new_gene1 == cross1[indexlist[index3[l]][0]][indexlist[index3[l]][1]]:
        new_gene1 = random.choice(limit[indexlist[index3[l]][1]])
      else:
        cross1[indexlist[index3[l]][0]][indexlist[index3[l]][1]] = new_gene1

  for l in range(len(index4)):
    if indexlist[index4[l]][0] == 0 and indexlist[index4[l]][1] == 0:
      new_gene2 = random.choice(limit[indexlist[index4[l]][1]][1:])
      while new_gene2 == cross2[indexlist[index4[l]][0]][indexlist[index4[l]][1]]:
        new_gene2 = random.choice(limit[indexlist[index4[l]][1]][1:])
      else:
        cross2[indexlist[index4[l]][0]][indexlist[index4[l]][1]] = new_gene2
    else:
      new_gene2 = random.choice(limit[indexlist[index4[l]][1]])
      while new_gene2 == cross2[indexlist[index4[l]][0]][indexlist[index4[l]][1]]:
        new_gene2 = random.choice(limit[indexlist[index4[l]][1]])
      else:
        cross2[indexlist[index4[l]][0]][indexlist[index4[l]][1]] = new_gene2

  return cross1, cross2

def population_size(population,size):
  population = population[:size]
  return population

def GA(num_cromosome, num_generation, cross_prob, mutate_prob):
  #populasi awal
  population = population_init(num_cromosome)

  for j in range(len(population)):
    fit = fitnesscnn(population[j][0], X_train, Y_train, x_test, y_test)
    population[j][1] = fit

  population = sorted(population, key = lambda x: (1/(x[1]), (x[0][0][0]*x[0][0][1]+x[0][1][0]*x[0][1][1]+x[0][2][0]*x[0][2][1])))

  popp = []
  popp.append([population])
  
  for i in range(num_generation):
    print('Generation: ', i+1)
    print("==========================================================================================================")

    new_population = []
    for _ in range(int((len(population) * cross_prob) / 2)):
      cross1, cross2 = cross(population)
      new1, new2 = mutate(cross1, cross2, mutate_prob)

      while new1.tolist() == new2.tolist():
        new1, new2 = mutate(cross1, cross2, mutate_prob)
      else:
        for j in range(len(population)):
          while new1.tolist() == population[j][0] or new2.tolist() == population[j][0]:
            new1, new2 = mutate(cross1, cross2, mutate_prob)
        
        for m in range(len(new_population)):
          while new1.tolist() == new_population[m][0] or new2.tolist() == new_population[m][0]:
            new1, new2 = mutate(cross1, cross2, mutate_prob)
        else:
          new_population.append([cross1.tolist(), 0])
          new_population.append([cross2.tolist(), 0])

    new_population = np.array(new_population)

    for k in range(len(new_population)):
      new_fit = fitnesscnn(new_population[k][0], X_train, Y_train, x_test, y_test)
      new_population[k][1] = new_fit

    population = np.append(population, new_population, axis = 0)

    #menghitung nilai fitness(akurasi) dari tiap individu
    population = sorted(population, key = lambda x: (1/(x[1]), (x[0][0][0]*x[0][0][1]+x[0][1][0]*x[0][1][1]+x[0][2][0]*x[0][2][1])))
    
    #elitism
    population = population[:num_cromosome]
    
    popp.append([population])

  else:
    print("done")

  return popp

data = open('M14-splitted.pkl','rb')
dataset = pickle.load(data)
data.close()

X_train = dataset[0]
Y_train = dataset[1]
x_test = dataset[2]
y_test = dataset[3]

popu2 = GA(num_cromosome = 10, num_generation = 20, cross_prob = 0.4, mutate_prob = 2/6)

for i in range(len(popu2)):
  for j in range(len(popu2[i])):
    for k in range(len(popu2[i][j])):
      print(popu2[i][j][k][0], "\tfitness: ", popu2[i][j][k][1] * 100, "%")
  print("===================================================================")

f = open("model/percobaan 2/M14/M14-2.txt", "wb")
pickle.dump(str(popu2),f)
f.close()
