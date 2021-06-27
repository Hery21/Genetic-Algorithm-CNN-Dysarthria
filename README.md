# Genetic-Algorithm-CNN-Dysarthria
Dysarthria is a speech disorder caused by motoric imparment. Dysarthric speakers generally also suffer from difficulty of moving other body muscles, therefore human or technology assistance is needed to communicate and doing activities. Speech recognition is preferred over other assisting technologies that require moving other body muscles to communicate such as keyboard.
Automatic speech recognition can be build using classification algorithm, one of which is Convolutional Neural Network (CNN). CNN is one of machine learning techniques that shows superior performance in the computer vision field. CNN also shows a good performance in speech recognition using spectogram as the speech representation. However, a good CNN architecture requires experties in the data and CNN designing. This problem can be overcome by using metaheuristic algorithm that searches most optimal CNN architecture from a search space. In this research, genetic algorithm is used to search for CNN architecture that is closest to its optimal.
This hybrid between CNN and genetic algorithm shows a better classification accuracy averaging in 96,19% compared to prior research that manually design the CNN averaging in 88,57%.

# Data
The data is collected from University of Illinois, the UA Speech Database.

# Preprocess
The data is preprocessed using Audacity to reduce the noise, normalize, and cut or added silence to equalize the length.

# Feature Extraction
The feature is extracted using Mel Frquency Cepstral Coefficient and its two derivatives. It is then stacked to make a three dimensional shape. The data is splitted into training and testing data in 2 : 1 manner.
![image](https://user-images.githubusercontent.com/45752762/123560413-250f6c00-d7cc-11eb-933d-0af537037388.png)

# Chromosome representation
![image](https://user-images.githubusercontent.com/45752762/123560447-5f790900-d7cc-11eb-8aa8-1c481e77324f.png)

# Crossover
![image](https://user-images.githubusercontent.com/45752762/123560464-7a4b7d80-d7cc-11eb-95ba-a51baaf88156.png)

# Mutation
![image](https://user-images.githubusercontent.com/45752762/123560469-7fa8c800-d7cc-11eb-8cde-9163daaffb7f.png)
