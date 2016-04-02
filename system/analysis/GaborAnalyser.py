import bob.ip.gabor
import bob.io.base.test_utils
from scipy import misc
import numpy as np
import glob
import datetime
from multiprocessing import Process, Manager
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')


CLASSES = {
    "Up": 0,
    "Down": 1,
    "Right": 2,
    "Left": 3,
    "Closed": 4,
    "Open": 5
}


def main():
    manager = Manager()

    real_jets = manager.dict()
    imag_jets = manager.dict()

    folders = glob.glob("../../data/train/*/")

    names = [target.split("/")[-2] for target in folders]
    for name in names:
        real_jets[name] = np.array([])
        imag_jets[name] = np.array([])

    processes = [Process(target=analyse, args=(folder, real_jets, imag_jets, )) for folder in folders]
    [p.start() for p in processes]
    [p.join() for p in processes]

    # Put wavelets as main index

    matrix_real = None
    matrix_imag = None
    print("Preparing Data")
    for name in names:
        real_jets[name] = real_jets[name]
        imag_jets[name] = imag_jets[name]
        if matrix_real is None:
            matrix_real = [dict() for i in range(len(real_jets[name]))]
            matrix_imag = [dict() for i in range(len(real_jets[name]))]
        # Name - Jet - Vector - Val
        # Jet1 - Names - Vector - Val
        for i in range(len(real_jets[name])):
            matrix_real[i][name] = real_jets[name][i]
            matrix_imag[i][name] = imag_jets[name][i]

    del real_jets
    del imag_jets

    predictions = manager.list()
    [predictions.append(list()) for i in range(len(matrix_real))]

    test_files = glob.glob("../../data/test/*")
    print("Starting Classifiers")
    processes = [Process(target=train_and_predict, args=(i, matrix_real[i], matrix_imag[i], test_files, predictions)) for i in range(len(matrix_real))]
    [p.start() for p in processes]
    [p.join() for p in processes]

    del matrix_real
    del matrix_imag

    test_file_predictions = [[0 for i in range(6)] for j in range(len(test_files))]
    for classifier in range(len(predictions)):
        for image in range(len(predictions[classifier])):
            prediction = predictions[classifier][image]
            test_file_predictions[image][prediction] += 1

    for predict in range(len(test_file_predictions)):
        print(test_files[predict] + " : " + ",".join([str(really) for really in test_file_predictions[predict]]))

    print("Analysis done")


def train_and_predict(jetId, jet_real, jet_imag, test_data, predictions):

    global CLASSES
    samples = list()
    classes = list()

    # 6 x N_IMAGES x 72
    # N_IMAGES x 144

    for key in jet_real:
        for i in range(len(jet_real[key])):
            samples.append(jet_real[key][i].tolist() + jet_imag[key][i].tolist())
            classes.append(CLASSES[key])

    classifier = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    classifier.fit(samples, classes)

    to_predict = list()

    for image_name in test_data:
        image = misc.imread(image_name, mode="L")
        gray_scale = misc.imresize(image, (63, 83))

        extractor = bob.ip.gabor.Graph((0, 0), (gray_scale.shape[0] - 1, gray_scale.shape[1] - 1), (8, 8))

        # perform Gabor wavelet transform on image
        gwt = bob.ip.gabor.Transform(number_of_scales=9)
        trafo_image = gwt(gray_scale)

        jets = extractor.extract(trafo_image)
        to_predict.append(jets[jetId].complex.real.tolist() + jets[jetId].complex.imag.tolist())

    predictions[jetId] = classifier.predict(to_predict)


def analyse(target, real, imag):

    cnter = 0
    sub_real = list()
    sub_imag = list()

    target_name = target.split("/")[-2]

    for name in glob.glob(target + "*"):
        image = misc.imread(name, mode="L")
        gray_scale = misc.imresize(image, (63, 83))

        extractor = bob.ip.gabor.Graph((0, 0), (gray_scale.shape[0]-1, gray_scale.shape[1]-1), (8, 8))

        # perform Gabor wavelet transform on image
        gwt = bob.ip.gabor.Transform(number_of_scales=9)
        trafo_image = gwt(gray_scale)

        jets = extractor.extract(trafo_image)
        sub_real.append(np.array([x.complex.real for x in jets]))
        sub_imag.append(np.array([x.complex.imag for x in jets]))

        if cnter % 250 == 0:
            now = datetime.datetime.now()
            print(str(now.hour)+":"+str(now.minute)+":"+str(now.second)+"  -  " + target_name + "  -  "+str(cnter))

        cnter += 1

    print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "  -  " + target_name + "  -  Gathering Statistics")

    real_jets = [None for i in range(len(sub_real[0]))]
    imag_jets = [None for i in range(len(sub_imag[0]))]

    for i in range(len(sub_real)):
        for j in range(len(sub_real[i])):

            if real_jets[j] is None:
                real_jets[j] = list()
                imag_jets[j] = list()

            real_jets[j].append(sub_real[i][j])
            imag_jets[j].append(sub_imag[i][j])

    real[target_name] = real_jets
    imag[target_name] = imag_jets

    print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "  -  " + target_name + "  -  Done!")


def print_to_file(prefix, data, names):
    header = "".join([name+"\tstddev\t" for name in names]) + "\n"

    matrix = None

    with open("../../dump/" + prefix, "w") as f:
        for name in names:
            if matrix is None:
                matrix = [[data[name][i][j].tolist() for j in range(len(data[name][i]))] for i in range(len(data[name]))]
            else:
                [[matrix[i][j].extend(data[name][i][j]) for j in range(len(data[name][i]))] for i in range(len(data[name]))]

        for i in range(len(matrix)):
            f.write("Jet-" + str(i) + "\n")
            f.write(header)
            for vector in matrix[i]:
                f.write("\t".join(str(v) for v in vector) + "\n")
            f.write("\n\n")

if __name__ == "__main__":
    main()
