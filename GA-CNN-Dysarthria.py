data = open('M09_186*13.pkl','rb')
dataset = pickle.load(data)
data.close()

full = np.array(list(dataset.items()))

images = []
labels = []

for i,j in full:
  image = np.array(j.transpose(2,1,0))
  label = np.array(int(i[-8]))

  images.append(image)
  labels.append(label)

def fitnesscnn(individual, images, labels):
  seed = 42
  np.random.seed(seed)
  random.seed(seed)
  tf.random.set_seed(seed)
  (train_images, test_images, train_labels, test_labels) = train_test_split(images, labels, test_size=0.33, random_state = 21)

  train_images = np.array(train_images)
  test_images = np.array(test_images)

  train_labels = np.array(train_labels)
  test_labels = np.array(test_labels)

  train_mean = np.mean(train_images)
  train_stdev = np.std(train_images)
  train_images = (train_images - train_mean)/train_stdev
  test_images = (test_images - train_mean)/train_stdev

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
  
  opt = keras.optimizers.rmsprop(learning_rate=0.0001, decay=1e-6)
  model.summary()
  model.compile(
      loss='categorical_crossentropy',
      optimizer=opt,
      metrics=['accuracy'],
      )
  history = model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=20,
    validation_data=(test_images, to_categorical(test_labels)),
    )
  
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
    
  fitness = np.array(history.history['val_accuracy'][-1])
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

def crossmutate(population):
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

  limit = {0: [0,3,5,7],
           1: [5,10,12,15,20]
           }
  
  #memilih titik mutasi
  cromosome_point = np.random.choice(len(cross1), 2)
  gene_point = np.random.choice([0, 1], 2)

  if cromosome_point[0] == 0 and gene_point[0] == 0:
    new_gene1 = random.choice(limit[gene_point[0]][1:])
  else:
    new_gene1 = random.choice(limit[gene_point[0]])

  if cromosome_point[1] == 0 and gene_point[1] == 0:
    new_gene2 = random.choice(limit[gene_point[1]][1:])
  else:
    new_gene2 = random.choice(limit[gene_point[1]])

  #memastikan mutasi gen berbeda
  while new_gene1 == cross1[cromosome_point[0]][gene_point[0]]:
    if cromosome_point[0] == 0 and gene_point[0] == 0:
      new_gene1 = random.choice(limit[gene_point[0]][1:])
    else:
      new_gene1 = random.choice(limit[gene_point[0]])
  else:
    cross1[cromosome_point[0]][gene_point[0]] = new_gene1

  while new_gene2 == cross2[cromosome_point[1]][gene_point[1]]:
    if cromosome_point[1] == 0 and gene_point[1] == 0:
      new_gene2 = random.choice(limit[gene_point[1]][1:])
    else:
      new_gene2 = random.choice(limit[gene_point[1]])
  else:
    cross2[cromosome_point[1]][gene_point[1]] = new_gene2

  if cross1.tolist() != cross2.tolist():
    for i in range(len(population)):
      while cross1.tolist() == population[i][0] or cross2.tolist() == population[i][0]:
        return crossmutate(population)
      else:
        continue
    else:
      return cross1.tolist(), cross2.tolist()
  elif cross1.tolist() == cross2.tolist():
    return crossmutate(population)

def population_size(population,size):
  population = population[:size]
  return population

def GA():
  num_cromosome = 10
  generation = 1

  #populasi awal
  population = population_init(num_cromosome)

  for j in range(len(population)):
    fit = fitnesscnn(population[j][0], images, labels)
    population[j][1] = fit

  popp = []
  
  for i in range(50):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    #penambahan individu hasil crossover dan mutasi
    new1, new2 = crossmutate(population)
    fitnew1, fitnew2 = fitnesscnn(new1, images, labels), fitnesscnn(new2, images, labels)

    population = np.append(population, [np.array([new1, fitnew1])], axis = 0)
    population = np.append(population, [np.array([new2, fitnew2])], axis = 0)
    
    #menghitung nilai fitness(akurasi) dari tiap individu

    population = sorted(population, key = lambda x: (1/(x[1]), x[0][0][0]*x[0][0][1]+x[0][1][0]*x[0][1][1]+x[0][2][0]*x[0][2][1]))

    #elitism
    population = population_size(population, num_cromosome)
    
    popp.append([population])
    
    print('Generation: ', generation)
    generation += 1
    
  else:
    print("done")
    
  return popp

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

popu2 = GA()