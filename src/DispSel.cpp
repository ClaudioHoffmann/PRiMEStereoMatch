/*---------------------------------------------------------------------------
   DispSel.cpp - Disparity Selection Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "DispSel.h"

DispSel::DispSel()
{
#ifdef DEBUG_APP
		std::cout <<  "Winner-Takes-All Disparity Selection." << std::endl;
#endif // DEBUG_APP
}
DispSel::~DispSel() {}

void DS_X(DS_X_TD t_data)
{
    //Matricies
	cv::Mat* costVol = t_data.costVol;
	cv::Mat* dispMap = t_data.dispMap;
    //Variables
	int y = t_data.y;
	int maxDis = t_data.maxDis;

	int wid = dispMap->cols;
	unsigned char* dispData = (unsigned char*) dispMap->ptr<unsigned char>(y);

	for(int x = 0; x < wid; ++x)
	{
		auto minCost = std::numeric_limits<float>::max();
		int minDis = 0;

		for(int d = 1; d < maxDis; ++d)
		{
			float* costData = (float*)costVol[d].ptr<float>(y);
			if(costData[x] < minCost)
			{
				minCost = costData[x];
				minDis = d;
			}
		}
		dispData[x] = minDis;
	}
}

int DispSel::CVSelect_thread(cv::Mat* costVol, const unsigned int maxDis, cv::Mat& dispMap, unsigned int threads)
{
    unsigned int hei = dispMap.rows;

	//Set up threads for x-loop
	std::vector<std::thread> DS_X_threads(hei);

    for(unsigned int level = 0; level <= hei/threads; ++level)
	{
        //Handle remainder if threads is not power of 2.
	    int block_size = (level < hei/threads) ? threads : (hei%threads);

	    for(int iter=0; iter < block_size; ++iter)
	    {
	        int d = level*threads + iter;
			DS_X_threads[d] = std::thread(DS_X, DS_X_TD{ costVol, &dispMap, d, maxDis });
	    }
        for(int iter=0; iter < block_size; ++iter)
	    {
	        int d = level*threads + iter;
			DS_X_threads[d].join();
        }
	}
	return 0;
}

int DispSel::CVSelect(cv::Mat* costVol, const unsigned int maxDis, cv::Mat& dispMap)
{
    unsigned int hei = dispMap.rows;
    unsigned int wid = dispMap.cols;

	#pragma omp parallel for
    for(unsigned int y = 0; y < hei; ++y)
    {
		for(unsigned int x = 0; x < wid; ++x)
		{
			auto minCost = std::numeric_limits<float>::max();
			int minDis = 0;

			for(unsigned int d = 1; d < maxDis; ++d)
			{
				float* costData = (float*)costVol[d].ptr<float>(y);
				if(costData[x] < minCost)
				{
					minCost = costData[x];
					minDis = d;
				}
			}
			dispMap.at<unsigned char>(y,x) = minDis;
		}
    }
    return 0;
}
