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

trainData = dict(rddFile.map(lambda file: ((userIdMapping[file[0]], businessIdMapping[file[1]]), float(file[2]))).collect())
userList = dict(rddFile.map(lambda file: (userIdMapping[file[0]], businessIdMapping[file[1]])).groupByKey().map(lambda file: (file[0], set(file[1]))).collect())
businessList = dict(rddFile.map(lambda file: (businessIdMapping[file[1]], userIdMapping[file[0]])).groupByKey().map(lambda file: (file[0], set(file[1]))).collect())

userAvg = dict()

for i in userList:
    sum = 0
    cnt = 0
    for j in userList[i]:
        sum += trainData[(i, j)]
        cnt += 1
    userAvg[i] = float(sum/cnt)

def predictScore(test):

    testUser = test[0][0]
    testBusiness = test[0][1]

    if (testUser not in userList) or (testBusiness not in businessList):
        x = ((testUser, testBusiness), 3.0)
        return x

    if testBusiness in userList[testUser]:
        x = ((testUser, testBusiness), trainData[(testUser, testBusiness)])
        return x

    candidateUsers = set(businessList[testBusiness])
    pnumerator = 0.0
    pdenum = 0.0

    for i in candidateUsers:

        candidateItems = list(set(userList[i]).intersection(set(userList[testUser])))
        if len(candidateItems) == 0:
            continue
        numerator = 0.0
        denum1 = 0.0
        denum2 = 0.0
        j = 0
        while j < len(candidateItems):
            numerator += ((trainData[(i,candidateItems[j])]-userAvg[i])*(trainData[(testUser,candidateItems[j])]-userAvg[testUser]))
            denum1 += ((trainData[(i,candidateItems[j])]-userAvg[i])**2)
            denum2 += ((trainData[(testUser,candidateItems[j])]-userAvg[testUser])**2)
            j += 1

        if denum1 == 0 or denum2 == 0:
            w = 0
        else:
            w = (numerator)/((denum1**0.5)*(denum2**0.5))
        pnumerator += (trainData[(i, testBusiness)]-userAvg[i]) * w
        pdenum += abs(w)

    if pdenum == 0:
        predictedrating = userAvg[testUser]
    else:
        predictedrating = userAvg[testUser] + (pnumerator/pdenum)

    if predictedrating > 5 or predictedrating <= 0:
        predictedrating = 3

    x = ((testUser, testBusiness), predictedrating)

    return x

ans = mappedTestData.map(predictScore)
ansf = ans.collect()

outputFile = open(str(sys.argv[4]), "w")
outputFile.write("user_id, business_id, prediction\n")
for i in ansf:
    outputFile.write(
        str(inverseUserIdMapping[i[0][0]]) + "," + str(inverseBusinessIdMapping[i[0][1]]) + "," + str(i[1]) + "\n")