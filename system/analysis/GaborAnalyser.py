import bob.ip.gabor
import bob.io.base.test_utils
from scipy import misc
import numpy as np
import glob
import sys
import random
import datetime
import ctypes as c
from multiprocessing import Process, Array
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = "/home/nixon/PycharmProjects/SYSC5405Project"
PREDICTION_FILE_NAME = "predictions"
JETS = 88
VECTORS = 72
MAX_IMAGES_PER_CLASS = None
CROSS_VALIDATION = None
DEBUG = True
PRODUCTION = True

CLASSES = {
    "Up": 0,
    "Down": 1,
    "Right": 2,
    "Left": 3,
    "Closed": 4,
    "Open": 5
}

CLASS_ID_LOOKUP = ["" for key in CLASSES]
for key in CLASSES:
    CLASS_ID_LOOKUP[CLASSES[key]] = key

CLASS_ID_LOOKUP_PRODUCTION = [key[0] for key in CLASS_ID_LOOKUP]


def train(features, classes):
    classifier = Pipeline([
        ('classification', QuadraticDiscriminantAnalysis())
    ])
    classifier.fit(features, classes)
    return classifier


def main():
    train_data = dict()
    if CROSS_VALIDATION is None and not PRODUCTION:
        for key in CLASSES:
            train_data[key] = glob.glob(BASE_PATH + "/data/train/" + key + "/*")
            if MAX_IMAGES_PER_CLASS is not None and MAX_IMAGES_PER_CLASS < len(train_data[key]):
                train_data[key] = train_data[key][:MAX_IMAGES_PER_CLASS]
        test_data = glob.glob(BASE_PATH + "/data/test/*")
        test_data_classes = [2, 4, 1, 0, 3, 5]
        predictions = gabor_analyse(train_data, test_data)
        matrix, tie = precision_assesment(predictions, test_data_classes)
        print('-' * 60)
        print("Test Data")
        print_confusion_matrix(matrix, tie)
        print('-' * 60)
    elif not PRODUCTION:
        total_data = dict()
        for key in CLASSES:
            total_data[key] = glob.glob(BASE_PATH + "/data/train/"+key+"/*")
            if MAX_IMAGES_PER_CLASS is not None and MAX_IMAGES_PER_CLASS < len(total_data[key]):
                total_data[key] = total_data[key][:MAX_IMAGES_PER_CLASS]
            total_data[key] = slice_list(total_data[key], CROSS_VALIDATION)

        # noinspection PyTypeChecker
        for i in range(CROSS_VALIDATION):
            print('-' * 60)
            print("Cross Validation fold " + str(i))
            print('-' * 60)
            train_data = dict()
            test_data_classes = list()
            test_data = list()
            for key in CLASSES:
                train_data[key] = list()
                for data_slice in range(len(total_data[key])):
                    if data_slice != i:
                        train_data[key].extend(total_data[key][data_slice])
                    else:
                        test_data.extend(total_data[key][data_slice])
                        test_data_classes.extend([CLASSES[key]]*len(total_data[key][data_slice]))
            predictions = gabor_analyse(train_data, test_data)
            matrix, tie = precision_assesment(predictions, test_data_classes)
            print('-' * 60)
            print("Cross validation run " + str(i))
            print_confusion_matrix(matrix, tie)
            print('-' * 60)
    elif PRODUCTION:
        for key in CLASSES:
            train_data[key] = glob.glob(BASE_PATH + "/data/train/" + key + "/*")
            if MAX_IMAGES_PER_CLASS is not None and MAX_IMAGES_PER_CLASS < len(train_data[key]):
                train_data[key] = train_data[key][:MAX_IMAGES_PER_CLASS]

        test_data = [BASE_PATH + "/data/production/" + str(i) + ".png"
                     for i in range(len(glob.glob(BASE_PATH + "/data/production/*")))]
        predictions = gabor_analyse(train_data, test_data)
        write_predictions(predictions)


def write_predictions(predictions):
    print("Writing Predictions for " + str(len(predictions)) + " images")
    with open(BASE_PATH + "/dump/" + PREDICTION_FILE_NAME, "w") as pred_file:
        for i in range(len(predictions)):
            if len(predictions[i]) > 1:
                prediction = [random.choice(predictions[i])]
                print("Detected tie for image " + str(i) + "! Going with random pick " + str(prediction) +
                      " out for " + str(predictions[i]))
            else:
                prediction = predictions[i][0]
            pred_file.write(str(i)+": " + CLASS_ID_LOOKUP_PRODUCTION[prediction] + "\n")


def print_confusion_matrix(matrix, tie):
    print()
    print("Confusion Matrix")
    print()
    print("\t\t" + "\t".join(CLASS_ID_LOOKUP))
    correct = 0
    wrong = 0
    for i in range(len(matrix)):
        class_id = CLASS_ID_LOOKUP[i]
        print((class_id if class_id != "Up" else class_id + "\t") +
              "\t" + "\t".join([str(score) for score in matrix[i]]))
        for j in range(len(matrix[i])):
            if j == i:
                correct += matrix[i][j]
            else:
                wrong += matrix[i][j]
    print()
    print("Correct: " + str(correct))
    print("Wrong: " + str(wrong))
    print("Tie: " + str(tie))
    print("Total: " + str(correct+wrong+tie))
    print()


def precision_assesment(predictions, test_data_classes):
    tie = 0
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for j in range(len(predictions)):
        prediction = predictions[j]
        if len(prediction) > 1:
            tie += 1
        else:
            prediction = int(prediction[0])
            actual = int(test_data_classes[j])
            matrix[actual][prediction] += 1
    return matrix, tie


def slice_list(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def gabor_analyse(train_data, test_data):

    features_per_class = dict()

    print("Allocating Memory")
    for key in CLASSES:
        number_of_pictures = len(train_data[key])
        features_per_class[key] = np.frombuffer(Array(c.c_double, JETS*number_of_pictures*(2*VECTORS)).get_obj())
        features_per_class[key] = features_per_class[key].reshape((JETS, number_of_pictures, (2*VECTORS)))

    print("Processing images")
    processes = [Process(target=analyse, args=(CLASSES[key], train_data[key], features_per_class)) for key in CLASSES]
    [p.start() for p in processes]
    [p.join() for p in processes]

    print("Feature Selection")
    # NAME x JETS x SAMPLES x feature vector
    # 88 x (NAMExSAMPLES) x feature vector

    classes = list()
    features = list()

    for jet in range(len(features_per_class[next(iter(CLASSES.keys()))])):
        samples = np.row_stack([features_per_class[key][jet] for key in CLASSES])

        cl = list()
        for key in CLASSES:
            cl.extend([CLASSES[key]] * len(features_per_class[key][jet]))
        classes.append(cl)
        features.append(samples)

    del features_per_class

    print("Training Classifiers")
    classifiers = list()
    for feature_set in range(len(features)):
        classifiers.append(train(features[feature_set], classes[feature_set]))
        now = datetime.datetime.now()
        if DEBUG:
            print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) +
                  "  -  Done training classifier " + str(feature_set+1) + "/" + str(len(features)))
        else:
            sys.stdout.write("|")
    print()
    del features

    print("Testing Classifiers")

    winners, votes = predict(classifiers, test_data)
    if DEBUG:
        for image_id in range(len(test_data)):
            labeled = [(CLASS_ID_LOOKUP[classe] + '(' + str(votes[image_id][classe]) + ')')
                       for classe in range(len(votes[image_id]))]
            print('-' * 40)
            print("File: " + test_data[image_id])
            print("Winner: " + ", ".join([CLASS_ID_LOOKUP[win] for win in winners[image_id]]))
            print("Votes: " + ", ".join(labeled))

    print("Analysis done")
    return winners


def predict(classifiers, test_data):
    predictions_votes = list()
    winners = list()
    cnter = 0
    for image_name in test_data:
        jets = extract_features(image_name)
        votes = [0] * len(CLASSES)
        for jet in range(len(jets)):
            feature_vector = jets[jet]
            classe = int(classifiers[jet].predict(feature_vector)[0])
            prob = classifiers[jet].predict_proba(feature_vector)[0][classe]
            votes[classe] += 1 * prob
        predictions_votes.append(votes)
        m = max(votes)
        winners.append([i for i, j in enumerate(votes) if j == m])

        if cnter % 250 == 0 and DEBUG:
            now = datetime.datetime.now()
            print(str(now.hour)+":"+str(now.minute)+":"+str(now.second)+" - Classifying image "+str(cnter))
        cnter += 1
    now = datetime.datetime.now()
    print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + " - Done classifying " + str(cnter)+" images")
    return winners, predictions_votes


def analyse(classe_number, targets, features):
    cnter = 0
    classe = CLASS_ID_LOOKUP[classe_number]

    for name in targets:

        jets = extract_features(name)

        for jet in range(len(jets)):
            np.copyto(features[classe][jet][cnter], jets[jet])

        if cnter % 250 == 0 and DEBUG:
            now = datetime.datetime.now()
            print(str(now.hour)+":"+str(now.minute)+":"+str(now.second)+" - Analyzing " + classe + " - "+str(cnter))
        cnter += 1

    now = datetime.datetime.now()
    print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + " - Analyzing " +
          classe + " - Done! " + str(cnter) + " images processed")


def extract_features(image_name):
    image = misc.imread(image_name, mode="L")
    gray_scale = misc.imresize(image, (63, 83))

    extractor = bob.ip.gabor.Graph((0, 0), (gray_scale.shape[0] - 1, gray_scale.shape[1] - 1), (8, 8))

    # perform Gabor wavelet transform on image
    gwt = bob.ip.gabor.Transform(number_of_scales=9)
    trafo_image = gwt(gray_scale)

    # noinspection PyArgumentList
    jets = extractor.extract(trafo_image)
    return [np.concatenate((jets[jet].complex.real, jets[jet].complex.imag)) for jet in range(len(jets))]


if __name__ == "__main__":
    main()
