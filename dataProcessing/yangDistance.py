try:
    import cupy as xp
except ImportError:
    import numpy as xp
from scipy.stats import beta

def calcDifference(sample, aNeg, bNeg, aPos, bPos):
	negPDF = beta.pdf(sample, aNeg, bNeg)
	posPDF = beta.pdf(sample, aPos, bPos)
	pdfDiff = negPDF - posPDF
	pdfDiffNeg = xp.maximum(pdfDiff, xp.zeros_like(pdfDiff))
	pdfDiffPos = xp.maximum(-1 * pdfDiff, xp.zeros_like(pdfDiff))
	pdfMax = xp.maximum(negPDF, posPDF)
	return negPDF, posPDF, pdfDiffPos, pdfDiffNeg, pdfMax

def yangDistributionDifference(aNeg, bNeg, aPos, bPos, p=1):
	"""
	Eq. (7) from :

	Yang, R., Jiang, Y., Mathews, S. et al.
	Data Min Knowl Disc (2019) 33: 995.
	https://doi.org/10.1007/s10618-019-00622-6
	"""
	sampleSize = 1000
	negSample = xp.random.beta(aNeg, bNeg, sampleSize)
	posSample = xp.random.beta(aPos, bPos, sampleSize)
	negPDF_NEG, posPDF_NEG, pdfDiffPos_NEG, pdfDiffNeg_NEG, pdfMax_NEG = calcDifference(negSample, aNeg, bNeg, aPos, bPos)
	negPDF_POS, posPDF_POS, pdfDiffPos_POS, pdfDiffPOS_POS, pdfMax_POS = calcDifference(posSample, aNeg, bNeg, aPos, bPos)
	numerator1 = xp.mean(pdfDiffNeg_NEG / negPDF_NEG)
	numerator2 = xp.mean(pdfDiffPos_POS / posPDF_POS)
	sumVecs = xp.power(numerator1, xp.ones_like(numerator1) * p) + xp.power(numerator2, xp.ones_like(numerator2) * p)
	dPHat = xp.power(sumVecs, xp.ones_like(sumVecs) * (1/p))
	dTermNeg = (posPDF_NEG * 0.5) + (negPDF_NEG * 0.5)
	dTermPos = (posPDF_POS * 0.5) + (negPDF_POS * 0.5)
	denominator = (xp.sum(pdfMax_NEG / dTermNeg) + xp.sum(pdfMax_POS / dTermPos)) / (2 * sampleSize)
	return dPHat / denominator

def vectorPDistance(x, y, pExp):
    # binary vector selecting indices in which xi >= yi
    gtMask = (x >= y)
    # binary vector for selecting indices in which xi < yi
    ltMask = (x < y)

    gtSum = xp.sum(x[gtMask] - y[gtMask])**pExp
    ltSum = xp.sum(y[ltMask] - x[ltMask])**pExp
    return (gtSum + ltSum)**(1/pExp)

def yangVectorDistance(negativeVector, positiveVector, p=1):
    x = xp.array(negativeVector).reshape((-1,1))
    y = xp.array(positiveVector).reshape((-1,1))
    pExp = int(p)
    assert x.shape == y.shape, "x ({}) and y ({}) must be of the same shape".format(x.shape, y.shape)
    assert pExp > 0, "p must be an integer greater than 0"
    numerator = vectorPDistance(x,y,pExp)
    max_X_Y = xp.maximum(xp.absolute(x), xp.absolute(y))
    maxes = xp.maximum(max_X_Y, xp.absolute(x-y))
    return numerator / xp.sum(maxes)