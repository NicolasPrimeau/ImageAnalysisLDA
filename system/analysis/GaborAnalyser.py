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

    print_to_file("train-abs", statistics_abs, names)
    print_to_file("train-phases", statistics_phase, names)
    print("Analysis done")


def analyse(target, statistics_abs, statistics_phase):

    abs_mean = np.vectorize(lambda x: x.abs.mean())
    phase_mean = np.vectorize(lambda x: x.phase.mean())
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

        sub_abs.append(abs_mean(jets))
        sub_phase.append(phase_mean(jets))
        if cnter % 250 == 0:
            now = datetime.datetime.now()
            print(str(now.hour)+":"+str(now.minute)+":"+str(now.second)+"  -  " + target_name + "  -  "+str(cnter))

        cnter += 1

    print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "  -  " + target_name + "  -  Gathering Statistics")

    absolute = np.column_stack(sub_abs)
    phases = np.column_stack(sub_phase)
    statistics_abs[target_name] = [absolute.mean(axis=1), absolute.std(axis=1)]
    statistics_phase[target_name] = [phases.mean(axis=1), phases.std(axis=1)]
    print(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "  -  " + target_name + "  -  Done!")


def print_to_file(prefix, data, names):
    with open("../../dump/" + prefix, "w") as f:
        matrix = None
        for name in names:
            data[name] = data[name]
            f.write(name+"\t"+"stddev\t")
            if matrix is None:
                matrix = data[name]
            else:
                matrix = np.row_stack((matrix, data[name]))
        f.write("\n")

        for row in np.transpose(matrix):
            f.write("\t".join(row.astype(str))+"\n")


if __name__ == "__main__":
    main()