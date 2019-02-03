# based on the predicted score, generates the sorted recommendations for each business
import StringIO
def sortedRec(train_fname, validation_fname, pred_fname, sorted_out_fname):
    maxBusinessId = 0
    maxUserId = 0
    with open(train_fname) as f:
        contents = f.readlines()
    for line in contents:
        line = line.split(' ')
        businessId, userId = int(line[0]), int(line[1])
        maxBusinessId = max(maxBusinessId, businessId)
        maxUserId = max(maxUserId, userId)

    with open(validation_fname) as f:
        contents = f.readlines()
    for line in contents:
        line = line.split(' ')
        businessId, userId = int(line[0]), int(line[1])
        maxBusinessId = max(maxBusinessId, businessId)
        maxUserId = max(maxUserId, userId)

    with open(pred_fname) as f:
        contents = f.readlines()
    scores = []
    for line in contents:
        line = float(line.strip())
        scores.append(line)
    counter = 0;
    out_str = StringIO.StringIO()
    for businessId in range(maxBusinessId + 1):
        scoreDict = {}
        curr_str = StringIO.StringIO()
        for userId in range(maxUserId + 1):
            score = 0
            if counter < len(scores):
                score = scores[counter]
                counter += 1
            scoreDict[userId] = score
        sorted_score = sorted(scoreDict.items(), key = lambda x: x[1], reverse = True)
        for pair in sorted_score:
            curr_user, curr_score = pair[0], pair[1]
            curr_str.write(str(curr_user) + ':' + str(curr_score) + '\t')
        out_str.write(curr_str.getvalue().strip() + '\n')

    fout = open(sorted_out_fname, 'w')
    fout.write(out_str.getvalue())
    fout.close()


city = 'Tor'
train_fname = '/Users/rli02/Desktop/Yelp/' + city + '/' + city + '_training.txt'
validation_fname = '/Users/rli02/Desktop/Yelp/' + city + '/' + city + '_validation.txt'
pred_fname = 'data/' + city + '/data/' + city + '_prediction_probs.txt'
sorted_out_fname = '/Users/rli02/Desktop/Yelp/' + city + '/data/' + city + '_sorted_recommendation.txt'
sortedRec(train_fname, validation_fname, pred_fname, sorted_out_fname)


import numpy as np
# compute map score
def computeMAP(positionThres, train_fname, validation_fname, test_fname, sorted_rec_fname):
    businessUserCount = {}
    businessUserSet = {}
    with open(train_fname) as f:
        contents = f.readlines()
    for line in contents:
        line = line.strip(). split(' ')
        businessId, userId = int(line[0]), int(line[1])
        if businessId not in businessUserSet:
            businessUserSet[businessId] = set()
        businessUserSet[businessId].add(userId)
        businessUserCount[businessId] = businessUserCount.get(businessId, 0) + 1

    with open(validation_fname) as f:
        contents = f.readlines()
    for line in contents:
        line = line.strip().split(' ')
        businessId, userId = int(line[0]), int(line[1])
        if businessId not in businessUserSet:
            businessUserSet[businessId] = set()
        businessUserSet[businessId].add(userId)
        businessUserCount[businessId] = businessUserCount.get(businessId, 0) + 1

    testBusinessUserSet = {}
    with open(test_fname) as f:
        contents = f.readlines()
    for line in contents:
        line = line.strip(). split(' ')
        businessId, userId = int(line[0]), int(line[1])
        if businessId not in testBusinessUserSet:
            testBusinessUserSet[businessId] = set()
        testBusinessUserSet[businessId].add(userId)


    threshold = 0.1
    countThreshold = int(threshold * len(businessUserCount))
    topBusiness = sorted(businessUserCount.items(), key=lambda x: x[1], reverse = True)
    tailBusiness = sorted(businessUserCount.items(), key=lambda x: x[1])
    topBusinessSet = set()
    tailBusinessSet = set()
    count = 0
    for pair in topBusiness:
        business = pair[0]
        topBusinessSet.add(business)
        count += 1
        if count > countThreshold:
            break
    count = 0
    for pair in tailBusiness:
        business = pair[0]
        tailBusinessSet.add(business)
        count += 1
        if count > countThreshold:
            break


    f = open(sorted_rec_fname, 'r')
    business = 0
    APDict = {}
    AP_top_Dict = {}
    AP_tail_Dict = {}
    for line in f:
        array = line.strip().split()
        rankList = list()
        for scoreStr in array:
            scoreArray = scoreStr.split(':')
            userId = int(scoreArray[0])
            score = float(scoreArray[1])
            rankList.append((userId, score))
        if testBusinessUserSet.has_key(business):
            denominator = 1.0
            numerator = 1.0
            averageNumber = min(len(testBusinessUserSet[business]), positionThres)
            currentSum = []
            flag = False
            for pair in rankList:
                current_user = pair[0]
                if current_user in businessUserSet[business]:
                    continue
                if current_user in testBusinessUserSet[business]:
                    flag = True
                    currentSum.append(float(numerator)/float(denominator))
                    numerator += 1
                denominator += 1
            if flag == True:
                APDict[business] = np.mean(currentSum) #float(currentSum)/float(averageNumber)
                if business in topBusinessSet:
                    AP_top_Dict[business] = np.mean(currentSum) #float(currentSum)/float(averageNumber)
                elif business in tailBusinessSet:
                    AP_tail_Dict[business] = np.mean(currentSum) #float(currentSum)/float(averageNumber)
        business += 1
    f.close()
    MAP_mean = 0.0
    MAP_top = 0.0
    MAP_tail = 0.0
    for business in APDict:
        MAP_mean += float(APDict[business])
    for business in AP_top_Dict:
        MAP_top += float(AP_top_Dict[business])
    for business in AP_tail_Dict:
        MAP_tail += float(AP_tail_Dict[business])
    print "map for all, top, tail business", format(MAP_mean/len(APDict), '.4f'), format(MAP_top/len(AP_top_Dict), '.4f'), format(MAP_tail/len(AP_tail_Dict), '.4f')

positionThres = 10000
test_ground_truth_fname = '/Users/rli02/Desktop/Yelp/' + city + '/' + city + '_test.txt'
sorted_rec_fname = '/Users/rli02/Desktop/Yelp/' + city + '/data/' + city + '_sorted_recommendation.txt'
computeMAP(positionThres, train_fname, validation_fname, test_ground_truth_fname, sorted_rec_fname)
