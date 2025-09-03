def joinObjectArray(srcArr1, srcArr2, key):
	mergeArr = []
	for srcObj2 in srcArr2:
		def exist(existObj):
			mergeObj = {}
			if len(existObj) > 0:
				mergeObj = {**existObj[0], **srcObj2}
				mergeArr.append(mergeObj)
		exist(
			list(filter(lambda srcObj1: srcObj1[key] == srcObj2[key], srcArr1))
		)

	return mergeArr


def groupByField(arrSrc, field, applyKey, applyValue):
	temp = {}
	for item in arrSrc:
		key = applyKey(item[field])
		temp[key] = temp[key] if key in temp else []
		temp[key].append(applyValue(item))

	return temp
