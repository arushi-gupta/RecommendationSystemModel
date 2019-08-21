from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import sys

sc = SparkContext()
dataFile1 = sc.textFile(str(sys.argv[1]))
rddFile = dataFile1.map(lambda file: file.split(',')).filter(lambda file: file[0] != 'user_id').map(lambda file: (file[0], file[1], float(file[2])))

dataFile = sc.textFile(str(sys.argv[2]))
testDataFile = dataFile.map(lambda file: file.split(',')).filter(lambda file: file[0] != 'user_id').map(lambda file: (file[0], file[1], float(file[2])))


userIdMapping = dict()
businessIdMapping = dict()
usercnt = 1
businesscnt = 1
for i in rddFile.collect():
    if i[0] not in userIdMapping:
        userIdMapping[i[0]] = usercnt
        usercnt += 1
    if i[1] not in businessIdMapping:
        businessIdMapping[i[1]] = businesscnt
        businesscnt += 1

inverseUserIdMapping = dict()
inverseBusinessIdMapping = dict()

inverseUserIdMapping = {v: k for k, v in userIdMapping.items()}
inverseBusinessIdMapping = {v: k for k, v in businessIdMapping.items()}

userMappinglen = len(userIdMapping)
businessMappinglen = len(businessIdMapping)

for file in testDataFile.collect():
    if file[0] in userIdMapping and file[1] in businessIdMapping:
        continue
    elif file[0] in userIdMapping:
        businessMappinglen += 1
        businessIdMapping[file[1]] = businessMappinglen
        inverseBusinessIdMapping[businessMappinglen] = file[1]
    elif file[1] in businessIdMapping:
        userMappinglen += 1
        userIdMapping[file[0]] = userMappinglen
        inverseUserIdMapping[userMappinglen] = file[0]
    else:
        userMappinglen += 1
        businessMappinglen += 1
        userIdMapping[file[0]] = userMappinglen
        inverseUserIdMapping[userMappinglen] = file[0]
        businessIdMapping[file[1]] = businessMappinglen
        inverseBusinessIdMapping[businessMappinglen] = file[1]

mappedTestData = testDataFile.map(lambda file: ((userIdMapping[str(file[0])], businessIdMapping[str(file[1])]), float(file[2])))

mappedData = rddFile.map(lambda file: Rating(userIdMapping[str(file[0])], businessIdMapping[str(file[1])], file[2]))

rank = 10
numIterations = 10
model = ALS.train(mappedData, rank, numIterations, lambda_ = 0.25, nonnegative=True)
predictions = model.predictAll(mappedTestData.map(lambda file: (file[0][0], file[0][1]))).map(lambda file: ((file[0], file[1]), float(file[2])))
dataDifference = mappedTestData.subtractByKey(predictions).map(lambda file: (file[0], 3.0))

ans = sc.union([predictions, dataDifference])
ansf = ans.collect()

outputFile = open(str(sys.argv[4]), "w")
outputFile.write("user_id, business_id, prediction\n")
for i in ansf:
    outputFile.write(
        str(inverseUserIdMapping[i[0][0]]) + "," + str(inverseBusinessIdMapping[i[0][1]]) + "," + str(i[1]) + "\n")