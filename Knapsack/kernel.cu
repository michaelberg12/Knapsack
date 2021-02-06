#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>

__global__ void knapKernel(
    const uint16_t* maxWeight, const uint16_t* arraySize, 
    const uint16_t* valueD, const uint16_t* weightD, 
    unsigned long long int* victorId, uint16_t* victorValue, 
    unsigned long long int* offset)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int trueId = id + *offset;
    unsigned int sackWeight = 0;
    
    //check validity
    for (int a1 = 0; trueId; a1++) {
        if (trueId & 1) {
            sackWeight += (weightD[a1]);
        }
        trueId >>= 1;
    }
    //check against memory
    trueId = id + *offset;

    unsigned int sackValue = 0;
    if (sackWeight <= *maxWeight ) {
        for (int a1 = 0; trueId; a1++) {
            if (trueId & 1) {
                sackValue += (valueD[a1]);
            }
            trueId >>= 1;
        }
        if (sackValue > victorValue[id]) {
            victorValue[id] = sackValue;
            victorId[id] = id + *offset;
        }
    }
}

int main()
{

    uint16_t arraySize = 0;
    uint16_t maxWeight = 0;

    uint16_t* weightH = nullptr;
    uint16_t* valueH = nullptr;

    try {
        std::ifstream file("items.txt", std::ifstream::in);

        std::string line;
        for (int a1 = 0; std::getline(file, line); a1++)
        {
            switch(a1)
            {
                case 0:
                    maxWeight = (std::stoi(line));
                    break;
                case 1:
                    arraySize = std::stoi(line);
                    arraySize = static_cast<int>(arraySize);
                    weightH = (uint16_t*)malloc(sizeof(uint16_t) * arraySize);
                    valueH = (uint16_t*)malloc(sizeof(uint16_t) * arraySize);
                    break;
                default:
                    for (int a2 = 0; a2 < line.size(); a2++) {
                        if (line[a2] == ' ') {
                            std::size_t pos = line.find(" ");
                            weightH[a1 - 2] = std::stoi(line.substr(0, pos));
                            valueH[a1 - 2] = std::stoi(line.substr(pos));
                        }
                    }
            }
        }
        file.close();
    }
    catch (...) { //TODO: this is bad parctice so refactor
        std::cout << "Error reading file" << std::endl;
        return 0;
    }

    std::cout << arraySize << "\n" << maxWeight << std::endl;
    for (int a1 = 0; a1 < arraySize; a1++) {
        std::cout << "("<< weightH[a1] << " " << valueH[a1] << ")" << std::endl;
    }

    size_t size = arraySize * sizeof(uint16_t);

    cudaSetDevice(0);

    uint16_t* weightD;
    cudaMalloc(&weightD, size);
    uint16_t* valueD;
    cudaMalloc(&valueD, size);
    uint16_t* maxWeightD;
    cudaMalloc(&maxWeightD, sizeof(uint16_t));
    uint16_t* arraySizeD;
    cudaMalloc(&arraySizeD, sizeof(uint16_t));
    unsigned long long int* offsetD;
    cudaMalloc(&offsetD, sizeof(unsigned long long int));

    cudaMemcpy(weightD, weightH, size, cudaMemcpyHostToDevice);
    cudaMemcpy(valueD, valueH, size, cudaMemcpyHostToDevice);
    cudaMemcpy(maxWeightD, &maxWeight, sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arraySizeD, &arraySize, sizeof(uint16_t), cudaMemcpyHostToDevice);
 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 0-th device

    int blockSize = deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxBlocksPerMultiProcessor;
    unsigned long long int numThreads = pow(2.0, arraySize);
    int numBlocks = deviceProp.maxBlocksPerMultiProcessor * deviceProp.multiProcessorCount;

    uint16_t* victorValueD;
    cudaMalloc(&victorValueD, sizeof(uint16_t) * numBlocks * deviceProp.maxThreadsPerBlock);
    unsigned long long int* victorIdD;
    cudaMalloc(&victorIdD, sizeof(unsigned long long int) * numBlocks * deviceProp.maxThreadsPerBlock);

    //capable up to n=50 15 minutes run time 
    //the cpu finish the computation meaning that under ~ n=21 all computation is done by the cpu
    std::cout << numThreads << std::endl;
    for (unsigned long long int a1 = 0; a1 < numThreads; a1 += numBlocks * deviceProp.maxThreadsPerBlock) {
        
        cudaMemcpy(offsetD, &a1, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        knapKernel <<<numBlocks, blockSize >>> (maxWeightD, arraySizeD, valueD, weightD, victorIdD, victorValueD, offsetD);
    }
    
    //create host memory
    unsigned long long int*victorIdH = (unsigned long long int*)malloc (sizeof(unsigned long long int) * (numBlocks * deviceProp.maxThreadsPerBlock));
    uint16_t *victorValueH = (uint16_t*)malloc(sizeof(uint16_t) * (numBlocks * deviceProp.maxThreadsPerBlock));

    cudaMemcpy(victorIdH   , victorIdD,    sizeof(unsigned long long int) * numBlocks * deviceProp.maxThreadsPerBlock, cudaMemcpyDeviceToHost);
    cudaMemcpy(victorValueH, victorValueD, sizeof(uint16_t) * numBlocks * deviceProp.maxThreadsPerBlock, cudaMemcpyDeviceToHost);

    int highValue = 0;
    unsigned long long int highId = 0;

    for (int a1 = 0; a1 < numBlocks * deviceProp.maxThreadsPerBlock; a1++) {
        if (victorValueH[a1] > highValue) {
            highValue = victorValueH[a1];
            highId = victorIdH[a1];
        }
    }

    std::cout << "Value:" << highValue << " ID:" << highId << std::endl;
    for (int a1 = 0; highId; a1 ++) {
        if (highId & 1)
            printf("%d ", a1 + 1);

        highId >>= 1;
    }
    
    //free memory
    cudaFree(weightD);
    cudaFree(valueD);
    cudaFree(maxWeightD);
    cudaFree(arraySizeD);
    cudaFree(offsetD);

    //delete victorIdH;
    //delete victorValueH;


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //code from an example tutorial
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    //end code from an example tutorial

    return 0;
}


