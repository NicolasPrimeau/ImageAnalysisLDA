import bob.ip.gabor
import bob.io.base.test_utils
from scipy import misc
import numpy as np
import glob
import datetime
from multiprocessing import Process, Manager


def main():
    manager = Manager()

    statistics_abs = manager.dict()
    statistics_phase = manager.dict()

    folders = glob.glob("../../data/train/*/")

    names = [target.split("/")[-2] for target in folders]
    for name in names:
        statistics_abs[name] = np.array([])
        statistics_phase[name] = np.array([])

    processes = [Process(target=analyse, args=(folder, statistics_abs, statistics_phase, )) for folder in folders]
    [p.start() for p in processes]
    [p.join() for p in processes]

    for name in names:
        statistics_abs[name] = statistics_abs[name]
        statistics_phase[name] = statistics_phase[name]

    print_to_file("train-abs", statistics_abs, names)
    print_to_file("train-phases", statistics_phase, names)
    print("Analysis done")


def analyse(target, statistics_abs, statistics_phase):

    cnter = 0
    # load test image
    # image = bob.io.base.load(bob.io.base.test_utils.datafile("testimage.hdf5", 'bob.ip.gabor'))

    sub_abs = list()
    sub_phase = list()

    target_name = target.split("/")[-2]

    for name in glob.glob(target + "*"):
        image = misc.imread(name, mode="L")
        gray_scale = misc.imresize(image, (63, 83))

        extractor = bob.ip.gabor.Graph((0, 0), (gray_scale.shape[0]-1, gray_scale.shape[1]-1), (8, 8))

        # perform Gabor wavelet transform on image
        gwt = bob.ip.gabor.Transform(number_of_scales=9)
        trafo_image = gwt(gray_scale)

        jets = extractor.extract(trafo_image)
        sub_abs.append(np.array([x.abs for x in jets]))
        sub_phase.append(np.array([x.phase for x in jets]))

        if cnter % 250 == 0:
            now = datetime.datetime.now()
            print(str(now.hour)+":"+str(now.minute)+":"+str(now.second)+"  -  " + target_name + "  -  "+str(cnter))

        cnter += 1

    print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "  -  " + target_name + "  -  Gathering Statistics")

    abs_jets = [None for i in range(len(sub_abs[0]))]
    phase_jets = [None for i in range(len(sub_phase[0]))]

    for i in range(len(sub_abs)):
        for j in range(len(sub_abs[i])):

            if abs_jets[j] is None:
                abs_jets[j] = sub_abs[i][j]
                phase_jets[j] = sub_phase[i][j]
            else:
                abs_jets[j] = np.column_stack((abs_jets[j], sub_abs[i][j]))
                phase_jets[j] = np.column_stack((phase_jets[j], sub_phase[i][j]))

    statistics_abs[target_name] = np.array([np.column_stack((jet.mean(axis=1), jet.std(axis=1))) for jet in abs_jets])
    statistics_phase[target_name] = np.array([np.column_stack((jet.mean(axis=1), jet.std(axis=1))) for jet in phase_jets])

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
            #for i in range(len(data[name])):
            #    for j in range(len(data[name][i])):
            #        matrix[i][j].extend(data[name][i][j])

        for i in range(len(matrix)):
            f.write("Jet-" + str(i) + "\n")
            f.write(header)
            for vector in matrix[i]:
                f.write("\t".join(str(v) for v in vector) + "\n")
            f.write("\n\n")

if __name__ == "__main__":
    main()
