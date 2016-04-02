import bob.ip.gabor
import bob.io.base.test_utils
from scipy import misc
import numpy as np
import glob
import datetime
import ctypes as c
from multiprocessing import Process, Array
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = "/home/nixon/PycharmProjects/SYSC5405Project"
JETS = 88
VECTORS = 72
MAX_IMAGES_PER_CLASS = None
CROSS_VALIDATION = 5
DEBUG = False

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


def main():
    train_data = dict()
    if CROSS_VALIDATION is None:
        for key in CLASSES:
            train_data[key] = glob.glob(BASE_PATH + "/data/train/"+key+"/*")
            if MAX_IMAGES_PER_CLASS is not None and MAX_IMAGES_PER_CLASS < len(train_data[key]):
                train_data[key] = train_data[key][:MAX_IMAGES_PER_CLASS]
        test_data = glob.glob(BASE_PATH + "/data/test/*")
        gabor_analyse(train_data, test_data)
    else:
        total_data = dict()
        for key in CLASSES:
            total_data[key] = glob.glob(BASE_PATH + "/data/train/"+key+"/*")
            if MAX_IMAGES_PER_CLASS is not None and MAX_IMAGES_PER_CLASS < len(total_data[key]):
                total_data[key] = total_data[key][:MAX_IMAGES_PER_CLASS]
            total_data[key] = slice_list(total_data[key], CROSS_VALIDATION)

        for i in range(CROSS_VALIDATION):
            print('-' * 60)
            print("Cross Validation fold " + str(i))
            print('-' * 60 + '\n')
            train_data = dict()
            test_data_classes = list()
            test_data = list()
            for key in CLASSES:
                train_data[key] = list()
                for slice in range(len(total_data[key])):
                    if slice != i:
                        train_data[key].extend(total_data[key][slice])
                    else:
                        test_data.extend(total_data[key][slice])
                        test_data_classes.extend([CLASSES[key]]*len(total_data[key][slice]))
            predictions = gabor_analyse(train_data, test_data)
            precise, wrong, tie = precision_assesment(predictions, test_data_classes)
            print('-' * 60)
            print("Cross validation run " + str(i))
            print("Correct: " + str(precise))
            print("Wrong: " + str(wrong))
            print("Tie: " + str(tie))
            print("Total: " + str(len(test_data)))
            print('-' * 60)


def precision_assesment(predictions, test_data_classes):
    tie = 0
    precise = 0
    wrong = 0
    for j in range(len(test_data_classes)):
        prediction = predictions[j]
        if len(prediction) > 1:
            tie += 1
        else:
            prediction = prediction[0]
            actual = test_data_classes[j]
            if prediction == actual:
                precise += 1
            else:
                wrong += 1
    return precise, wrong, tie


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
        features_per_class[key] = np.frombuffer(Array(c.c_double, JETS*number_of_pictures*(2*VECTORS+1)).get_obj())
        features_per_class[key] = features_per_class[key].reshape((JETS, number_of_pictures, (2*VECTORS+1)))

    print("Processing images")
    processes = [Process(target=analyse, args=(CLASSES[key], train_data[key], features_per_class)) for key in CLASSES]
    [p.start() for p in processes]
    [p.join() for p in processes]

    print("Preparing Data")
    # NAME x JETS x SAMPLES x feature vector
    # 88 x (NAMExSAMPLES) x feature vector

    features = [np.row_stack(([features_per_class[key][jet] for key in CLASSES]))
                for jet in range(len(features_per_class[next(iter(CLASSES.keys()))]))]

    del features_per_class

    print("Training Classifiers")
    classifiers = list()
    for feature_set in range(len(features)):
        classifiers.append(train(features[feature_set]))
        now = datetime.datetime.now()
        if DEBUG:
            print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) +
                "  -  Done training classifier " + str(feature_set+1) + "/" + str(len(features)))
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


def train(features):
    samples = list()
    classes = list()
    np.random.shuffle(features)
    for feature in features:
        feature = feature.tolist()
        classe = feature.pop()
        samples.append(feature)
        classes.append(classe)

    classifier = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    classifier.fit(samples, classes)
    return classifier


def predict(classifiers, test_data):
    predictions_votes = list()
    winners = list()

    cnter = 0
    for image_name in test_data:
        image = misc.imread(image_name, mode="L")
        gray_scale = misc.imresize(image, (63, 83))

        extractor = bob.ip.gabor.Graph((0, 0), (gray_scale.shape[0] - 1, gray_scale.shape[1] - 1), (8, 8))

        # perform Gabor wavelet transform on image
        gwt = bob.ip.gabor.Transform(number_of_scales=9)
        trafo_image = gwt(gray_scale)

        # noinspection PyArgumentList
        jets = extractor.extract(trafo_image)
        votes = [0] * len(CLASSES)
        for jet in range(len(jets)):
            classe = classifiers[jet].predict([np.concatenate((jets[jet].complex.real, jets[jet].complex.imag))])[0]
            votes[int(classe)] += 1
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
    classe_number = [classe_number]

    for name in targets:
        image = misc.imread(name, mode="L")
        gray_scale = misc.imresize(image, (63, 83))

        extractor = bob.ip.gabor.Graph((0, 0), (gray_scale.shape[0]-1, gray_scale.shape[1]-1), (8, 8))

        # perform Gabor wavelet transform on image
        gwt = bob.ip.gabor.Transform(number_of_scales=9)
        trafo_image = gwt(gray_scale)

        # noinspection PyArgumentList
        jets = extractor.extract(trafo_image)

        for jet in range(len(jets)):
            np.copyto(features[classe][jet][cnter],
                      np.concatenate((jets[jet].complex.real, jets[jet].complex.imag, classe_number)))

        if cnter % 250 == 0 and DEBUG:
            now = datetime.datetime.now()
            print(str(now.hour)+":"+str(now.minute)+":"+str(now.second)+" - Analyzing " + classe + " - "+str(cnter))
        cnter += 1

    now = datetime.datetime.now()
    print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + " - Analyzing " +
          classe + " - Done! " + str(cnter) + " images processed")


if __name__ == "__main__":
    main()
