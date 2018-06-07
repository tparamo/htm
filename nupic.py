import os
import yaml
import r64_utils as r64
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.algorithms import anomaly_likelihood
from nupic.swarming import permutations_runner
import pandas as pd

import model_params

_FILE_PATH = os.path.normcase(os.path.dirname(__file__))
_PARAMS_PATH = os.path.join(_FILE_PATH, "data\\model_anomaly")
_ANOMALY_THRESHOLD = 0.9

def createModel():
  with open(_PARAMS_PATH, "r") as f:
    modelParams = yaml.safe_load(f)
  return ModelFactory.create(modelParams)

def importModel():
    return ModelFactory.create(model_params.MODEL_PARAMS)

def swarmModel(basedir, column):
    filename = os.path.join(_FILE_PATH, "data\\\swarm_params")
    model_params = permutations_runner.runWithJsonFile(filename, {'maxWorkers': 1, 'overwrite': True, 'verbosityCount':3}, column+"_swarm", basedir)
    model = ModelFactory.create(model_params)

def run(df, basedir, column):
    df.to_csv("C:\\Datos\\data.csv")
    model = swarmModel(basedir, column)
    model.enableInference({"predictedField": "VAR"})
    anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood()

    counter = 0
    results = []
    cerror = 0
    for index, row in df.iterrows():
        counter += 1
        if (counter % 1000 == 0):
            print "Read %i lines..." % counter

        modelInput = dict(zip(df.columns, row))
        result = model.run(modelInput)

        anomalyScore = result.inferences["anomalyScore"]
        bestPredictions = result.inferences["multiStepBestPredictions"]

        likelihood = anomalyLikelihood.anomalyProbability(modelInput["VAR"], anomalyScore, modelInput["time"])
        logLikelihood = anomalyLikelihood.computeLogLikelihood(likelihood)

        bestPredictions = result.inferences['multiStepBestPredictions']
        allPredictions = result.inferences['multiStepPredictions']
        oneStep = bestPredictions[1]
        # Confidence values are keyed by prediction value in multiStepPredictions.
        oneStepConfidence = allPredictions[1][oneStep]

        #Relative Percent Difference
        error = 0
        if oneStep!=0 or modelInput["VAR"]!=0:
            error = abs(modelInput["VAR"]-oneStep)/abs(max(modelInput["VAR"],oneStep)) * 100

        cerror += error

        results.append([modelInput["time"], modelInput["VAR"], oneStep, anomalyScore, likelihood, logLikelihood, error])


    results_filename = basedir + "//" + column + "_results.txt"
    with open(results_filename, 'w') as outfile:
        for result in results:
            for value in result:
                outfile.write(str(value)+ " ")
            outfile.write("\n")

    avg_error = cerror / counter
    summary = basedir + "//" + "summary.txt"
    with open(summary, 'w+') as sum:
        sum.write(column + " " + avg_error)

    model.save(basedir)

    return results


def mergeDataFrames():
    df1 = r64.prepareSequencialNormalicedData("C:\\Datos\\EDINCUBATOR\\dataset_1\\KBU1A1121650R02_NextGenDrive")
    df2 = r64.prepareSequencialNormalicedData("C:\\Datos\\EDINCUBATOR\\dataset_1\\KABHVS111110R01_NextGenDrive")
    df3 = r64.prepareSequencialNormalicedData("C:\\Datos\\EDINCUBATOR\\dataset_1\\KASTAL123860R01_NextGenDrive")

    total = pd.concat([df1, df2, df3], axis=0)
    aux = pd.to_datetime(pd.date_range('1/1/2018 00:00:00.006392', periods=len(total), freq='4ms')).values
    total['time'] = aux
    return total

def runDataSets():
    base_dir = "C:\\\Datos\\\EDINCUBATOR\\\dataset_1\\"
    suffix = "_NextGenDrive"
    robots = ["\KBU1A1121650R02", "\KABHVS111110R01", "\KASTAL123860R01"]
    for robot in robots:
        df = r64.prepareSequencialData(base_dir+robot+suffix)
        for column in df.columns[::-1]:
            if column != "time":
                invidual_df = pd.concat([df['time'], df[column]], axis=1)
                invidual_df.columns = ['time', 'VAR']
                run(invidual_df, base_dir+robot, column)


if __name__ == "__main__":
    #run(mergeDataFrames())
    runDataSets()