import matplotlib as plt
import nupic

def extractAnomalyData(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            words = line.split(" ")
            anomaly = words[4]
            likelyhood = words[5]
            data.append([anomaly, likelyhood])

    return data


def printAnomalyScore(df, anomalyFile, outputFile):

    fig, (ax1, ax2) = plt.subplots(num=2, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k', sharey=True)

    for dim in range(0, len(df.columns())):
        ax1.plot(range(0, len(df[:, dim])), df[:, dim], alpha=.8, labels="PCA"+str(dim+1))

    data = extractAnomalyData(anomalyFile)

    ax2.plot(range(0, len(df[:, 0])), df[:, 0], alpha=.8, labels="anomaly")
    ax2.plot(range(0, len(df[:, 1])), df[:, 1], alpha=.8, labels="likelihood")

    fig.save("")


if __name__ == "__main__":
    printAnomalyScore(nupic.mergeDataFrames(), "C:\\Datos\\EDINCUBATOR\\dataset_1\\results.txt", "C:\\Datos\\EDINCUBATOR\\dataset_1\\KBU1A1121650R02_KABHVS111110R01_KASTAL123860R01.txt")