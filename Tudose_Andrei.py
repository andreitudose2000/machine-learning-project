import numpy as np
from PIL import Image
import copy
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


# data_type -> 'train' / 'validation' / 'test'
def load_data(data_type):
    data_type = data_type.lower()

    # liste goale unde vom adauga datele
    ids = []
    samples = []
    labels = []

    # deschidem fisierul metadata corespunzator datelor pe care le citim
    path = 'data/' + data_type + '.txt'
    fin = open(path, mode='r', encoding='utf-8')

    for line in fin.readlines():
        if data_type != 'test':
            # pentru datele de antrenare si validare avem
            # denumirile si etichetele separate prin virgula
            id, label = line.split(',')
            labels.append(int(label[0]))
        else:
            # pentru datele de test avem doar denumirile
            id = line.replace("\n", "")
        ids.append(id)

        # deschidem imaginea folosing PIL.Image
        img_path = 'data/' + data_type + '/' + id
        img_obj = Image.open(img_path)

        # Convertim imaginea in numpy array unidimensional
        img = copy.deepcopy(np.asarray(img_obj).flatten())

        # important: inchidem obiectul Image
        img_obj.close()

        # adaugam imaginea ca numpy array la lista de sample-uri
        samples.append(img)

    # folosim StandardScaler din sklearn.preprocessing pentru
    # standardizarea datelor (creste performanta clasificatorului)
    sc = StandardScaler()
    samples = sc.fit_transform(samples)
    # Nota: La Naive Nayes nu putem standardiza datele pentru ca ia valori > 0

    # afisam media si deviatia standard
    print(sc.mean_)
    print(sc.scale_)

    # ids = np.array(ids)
    # transformam listele in numpy array-uri (mai eficiente ca timp de rulare
    # decat listele de Python + ofera mai multe functionalitati)
    samples = np.array(samples)
    labels = np.array(labels)

    return ids, samples, labels


# incarcam datele de antrenare
train_ids, train_samples, train_labels = load_data('train')

# incarcam datele de validare
validation_ids, validation_samples, validation_labels = load_data('validation')

# incarcam datele de testare
test_ids, test_samples, test_labels = load_data('test')


""" # Naive Bayes - 39.1%
num_bins = 9
nb_bins = np.linspace(start=0, stop=255, num=num_bins)


def values_to_bins(matrix, bins):
    matrix = np.digitize(matrix, bins)
    return matrix - 1


train_features_to_bins = values_to_bins(train_samples, nb_bins)
validation_features_to_bins = values_to_bins(validation_samples, nb_bins)

model = MultinomialNB()
model.fit(train_features_to_bins, train_labels)
predicted_validation_labels = model.predict(validation_features_to_bins)
accuracy = accuracy_score(validation_labels, predicted_validation_labels)
print(accuracy)
"""


""" # KNN - 57.7%
model = KNeighborsClassifier(n_neighbors=10)
model.fit(train_samples, train_labels)
predictions_knn = model.predict(validation_samples)
accuracy = accuracy_score(validation_labels, predictions_knn)
print(accuracy)
"""


""""# SVM - 76.1%
model = svm.SVC()
model.fit(train_samples, train_labels)
predicted = model.predict(validation_samples)
accuracy = accuracy_score(validation_labels, predicted)
print(accuracy)
"""


"""# RandomForest - 61.1%
model = RandomForestClassifier(n_estimators=100, random_state=9)
model.fit(train_samples, train_labels)
predicted = model.predict(validation_samples)
accuracy = accuracy_score(validation_labels, predicted)
print(accuracy)
"""


# Neural network

# am creat un clasificator cu hiperparametri obtinuti prin mai multe incercari
model = MLPClassifier(hidden_layer_sizes=668, max_iter=500)

# antrenez pe datele de train
model.fit(train_samples, train_labels)

# prezic label-urile pentru datele de validare
predicted = model.predict(validation_samples)

# calculez si afisez acuratetea predictiei folosing accuracy_score din sklearn.metrics
accuracy = accuracy_score(validation_labels, predicted)
print(accuracy)


# prezic label-urile pentru datele de test
test_labels = model.predict(test_samples)


# deschid un nou csv pentru a scrie submisia
wr = open("submission.csv", 'w')

# scriu antetul (primul rand)
wr.write("id,label" + "\n")

# pentru fiecare imagine de test scriu id-ul si labelul prezis de model
pairs = zip(test_ids, test_labels)
for id, prediction in pairs:
    wr.write(id + ',' + str(int(prediction)) + '\n')

# afisez matricea de confuzie folosind implementarea ei din sklearn.metrics
cm = confusion_matrix(validation_labels, predicted)
print(cm)
