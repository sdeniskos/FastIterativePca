// FastIterativePca.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <boost/array.hpp>
#include <assert.h>
#include <iostream>

//you need boost to compile
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/shared_ptr.hpp>
 

#pragma region ServiceFunctions
template<typename Collection>
__inline void normalizeL2(Collection& collection)
{
	typedef typename Collection::value_type value_type;
	value_type summ = std::accumulate(collection.begin(), collection.end(), (value_type)0, [](value_type & a, value_type b)->value_type{a = a + sqr(b); return a; });
	summ = sqrt(summ);
	if (summ < std::numeric_limits<value_type>::epsilon())
		return;
	std::for_each(collection.begin(), collection.end(), [summ](value_type& val)
	{
		val /= summ;
	});
}



template<typename OutArg, typename VecIn1, typename VecIn2>
__inline OutArg dotProduct(const VecIn1& a, const VecIn2& b)
{
	OutArg prod = 0;
	for (int i = 0; i < (int)a.size(); i++)
	{
		prod += static_cast<OutArg>(a[i]) * static_cast<OutArg>(b[i]);
	}
	return prod;
}
template<typename Type>
__inline Type sqr(Type t)
{
	return t * t;
}
double selectRandom(double a, double b, int range)
{
	return a + (b - a) *(rand() % range) / (range - 1);
}




struct NormalRandomGenerator
{
public:
	typedef boost::normal_distribution<double> NormalDistribution;
	typedef boost::variate_generator<boost::mt19937&, NormalDistribution > VarGen;
	typedef boost::shared_ptr<VarGen> VarGenPtr;
public:
	NormalRandomGenerator() :
		m_normalDist(0.0, 1.0)
	{
		m_varNor.reset(new VarGen(m_rng, m_normalDist));
	}
	void setParams(double  middle, double sigma)
	{
		m_normalDist = boost::normal_distribution<double>(middle, sigma);
		m_varNor.reset(new VarGen(m_rng, m_normalDist));
	}
	double generate()
	{
  
		return (*m_varNor)();
	}
private:
	NormalDistribution m_normalDist;
	VarGenPtr m_varNor;
	boost::mt19937 m_rng;
};
#pragma endregion ServiceFunctions



template<int dim>
struct FastIterativePcaSelector
{
	typedef boost::array<float, dim> Element;
	typedef std::vector<Element> Elements;
	typedef std::vector<float> Weights;

	Element getFirstPC(const Elements& elements, const Weights& weights, int numIterations, const Elements& prevBazis)
	{
		//sds we maximze projections on space with size 2
		std::vector<Element> currentVecs(2);
		for (int i1 = 0; i1 < 2; i1++)
		{
			if (i1 == 0)
				currentVecs[i1] = selectRandomVec(prevBazis);
			else
			{
				currentVecs[i1] = getRandomOrthogonalVec(currentVecs[0], prevBazis);
			}
		}
		double bestSigma = 0;
		for (int i1 = 0; i1 < numIterations; i1++)
		{
			Element currentApproximation = getApproximation(currentVecs.data(), elements, weights);
			Element newVec = getRandomOrthogonalVec(currentApproximation, prevBazis);
			currentVecs[0] = currentApproximation;
			currentVecs[1] = newVec;
			double sigma = calcSigma(elements, weights, currentApproximation);
			//stop criterion -- stabilization
			if (abs(sigma - bestSigma) < sigma * 0.00000001)
				break;

			std::cout << "num component " << prevBazis.size() << " " << "current sigma " << sigma << std::endl;
		}
		return currentVecs[0];
	}
	static double calcSigma(const Elements& elements, const Weights& weights, const Element& element)
	{
		double sigma = 0;
		for (int i1 = 0; i1 < (int)elements.size(); i1++)
		{
			sigma += sqr(dotProduct<float>(elements[i1], element))* weights[i1];
		}
		return sigma;
	}
private:
	
	static Element selectRandomVec(const Elements& prevBazis)
	{
		//select random vector, orthogonal to previous got bazis
		Element result;
		float dp = 0;
		

		//try new vector until it has some orthogonal part to previous bazis
		do
		{
			for (int i1 = 0; i1 < (int)result.size(); i1++)
			{
				result[i1] = (float)selectRandom(-1.0f, 1.0f, 1000);
			}
			for (int i1 = 0; i1 < (int)prevBazis.size(); i1++)
			{
				float dp = dotProduct<float>(result, prevBazis[i1]);
				for (int i2 = 0; i2 < (int)result.size(); i2++)
				{
					result[i2] = result[i2] - prevBazis[i1][i2] * dp;
				}
			}
			normalizeL2(result);
			dp = dotProduct<float>(result, result);
		} while (dp < 0.99);
		return result;
	}
	static Element getRandomOrthogonalVec(const Element& vec, const Elements& prevBazis)
	{
		
		Element result;
		float dp = 0;
		//try new vector until it has some orthogonal part to previous part and current best vector (vec) 
		do
		{
			result = selectRandomVec(prevBazis);
			dp = dotProduct<float>(result, vec);
			for (int i1 = 0; i1 < (int)result.size(); i1++)
			{
				result[i1] = result[i1] - vec[i1] * dp;
			}
			normalizeL2(result);
			dp = dotProduct<float>(result, result);
		} while (dp < 0.5f);
		return result;
	}



	Element getApproximation(const Element* approximationStack, const Elements& elements, const Weights& weights)
	{
		/*
		Maximize decomposition to R2 space spanned on two vectors .
		2 Plum -- you could use int D(w) * H(w) dw instead
		*/
		
		//fill covariation matrix
		double matrix[4] = { 0., 0., 0., 0. };
		for (int i0 = 0; i0 < (int)weights.size(); i0++)
		{
			float dps[2] = { dotProduct<float>(approximationStack[0], elements[i0]), dotProduct<float>(approximationStack[1], elements[i0]) };
			for (int i1 = 0; i1 < 2; i1++)
			{
				for (int i2 = 0; i2 < 2; i2++)
				{
					matrix[i1 * 2 + i2] += dps[i1] * dps[i2] * weights[i0];
				}
			}
		}

		//calc eigen vectors/values
		double offset = (matrix[0] + matrix[3]);
		double d = std::max(sqr(matrix[0] - matrix[3]) + 4 * (matrix[1] * matrix[2]), 0.);
		d = sqrt(d);
		double eV = offset < 0 ? offset / 2.0 - d / 2.0 : offset / 2.0 + d / 2.0;
		float coeffs[2];
		if (abs(matrix[0] - eV) > abs(matrix[3] - eV))
		{
			coeffs[0] = static_cast<float>(matrix[1] / (matrix[0] - eV));
			coeffs[1] = -1.f;
		}
		else
		{
			coeffs[0] = -1.f;
			coeffs[1] = static_cast<float>(matrix[2] / (matrix[3] - eV));
		}
		//make decomposition
		Element result;
		for (int i1 = 0; i1 < (int)approximationStack[0].size(); i1++)
		{
			result[i1] = approximationStack[0][i1] * coeffs[0] + approximationStack[1][i1] * coeffs[1];
		}
		normalizeL2(result);
		return result;
	}







};
//test function shows, how to use
struct FastIterativePcaSelectorTest
{
	static const int numDimensions = 10000;
	typedef FastIterativePcaSelector<numDimensions> FiPC;
	typedef FiPC::Element Element;
	typedef FiPC::Elements Elements;
	void test()
	{
		std::vector<double> sigmas(numDimensions);
		std::vector<boost::shared_ptr<NormalRandomGenerator> > normalDistributions(numDimensions);
		//generate some random distribution
		for (int i1 = 0; i1 < (int)sigmas.size(); i1++)
		{
			sigmas[i1] = 0.2 + (rand() & 0xff)*(200.0 - 0.2) / 255.0;
			normalDistributions[i1].reset(new NormalRandomGenerator());
			normalDistributions[i1]->setParams(0, sigmas[i1]);
		}
		int maxElt = std::max_element(sigmas.begin(), sigmas.end()) - sigmas.begin();
		int numElements = 200;
		std::vector<Element> elements(numElements);
		//assign all weights to 1
		std::vector<float> weights(numElements);
		for (int i1 = 0; i1 < (int)elements.size(); i1++)
		{
			for (int i2 = 0; i2 < numDimensions; i2++)
			{
				elements[i1][i2] = (float)normalDistributions[i2]->generate();
			}
			weights[i1] = 1.0;
		}
		//calc reference sigma along main axis
		Element bestElt;
		std::fill(bestElt.begin(), bestElt.end(), 0.f);
		bestElt[maxElt] = 1.0;



		double bestSigma = FiPC().calcSigma(elements, weights, bestElt);
		std::cout << bestSigma << std::endl;
		Elements bazis;
		Element middle;
		std::fill(middle.begin(), middle.end(), 1.0f);
		normalizeL2(middle);
		//insert middle in bazis, to maximize only zero mean vectors
		bazis.push_back(middle);
		static const int numComponents = 10;
		boost::shared_ptr<FiPC> fipcPtr(new FiPC());
		{
			for (int i1 = 0; i1 < numComponents; i1++)
			{
				Element comp = fipcPtr->getFirstPC(elements, weights, numDimensions, bazis);
				bazis.push_back(comp);
			}
			int j = 0;
		}

	}
private:


};

int _tmain(int argc, _TCHAR* argv[])
{
	FastIterativePcaSelectorTest().test();
	return 0;
}

