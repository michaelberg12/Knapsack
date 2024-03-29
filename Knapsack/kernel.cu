﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <chrono>
#include <atomic>


__global__ void knapKernel(
    const uint16_t* maxWeight, const uint16_t* arraySize,
    const uint16_t* valueD, const uint16_t* weightD,
    unsigned long long int* victorId, uint16_t* victorValue,
    unsigned long long int* offset, int* memSize, unsigned long long int runLength)
{
    unsigned long long int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int id = threadId + *offset;
    unsigned int sackWeight = 0;


    if (id <= runLength) {
        if (*offset == 0) {
            victorValue[threadId] = 0;
            victorId[threadId] = 0;
        }
        //check validity
        unsigned long long int trueIdLoc = id;
        for (int a1 = 0; trueIdLoc; a1++) {
            if (trueIdLoc & 1) {
                sackWeight += (weightD[a1]);
            }
            trueIdLoc >>= 1;
        }

        //check against memory
        unsigned int sackValue = 0;
        if (sackWeight <= *maxWeight) {
            trueIdLoc = id;
            for (int a1 = 0; trueIdLoc; a1++) {
                if (trueIdLoc & 1) {
                    sackValue += (valueD[a1]);
                }
                trueIdLoc >>= 1;
            }
            if (sackValue > victorValue[threadId]) {
                victorValue[threadId] = sackValue;
                victorId[threadId] = id;
            }
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
    int* memSizeD;
    cudaMalloc(&memSizeD, sizeof(int));

    cudaMemcpy(weightD, weightH, size, cudaMemcpyHostToDevice);
    cudaMemcpy(valueD, valueH, size, cudaMemcpyHostToDevice);
    cudaMemcpy(maxWeightD, &maxWeight, sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arraySizeD, &arraySize, sizeof(uint16_t), cudaMemcpyHostToDevice);
 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 0-th device

    //determin block size
    int blockSize = deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxBlocksPerMultiProcessor;
    unsigned long long int runLength = pow(2.0, arraySize);

    



    /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
    int numBlocksPerSm = 0;
    // Number of threads my_kernel will be launched with
    int numThreads = 128;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, knapKernel, numThreads, 0);


    //determin grid size
    int memSize = numBlocksPerSm * deviceProp.multiProcessorCount * numThreads;

    uint16_t* victorValueD;
    cudaMalloc(&victorValueD, sizeof(uint16_t) * memSize);
    unsigned long long int* victorIdD;
    cudaMalloc(&victorIdD, sizeof(unsigned long long int) * memSize);
    //copy number of threads
    
    //copy memory size
    cudaMemcpy(memSizeD, &memSize, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

    int incriment = deviceProp.multiProcessorCount * numBlocksPerSm * numThreads;

    for (unsigned long long int offset = 0; offset < runLength; offset += incriment) {
        cudaDeviceSynchronize();
        cudaMemcpy(offsetD, &offset, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        knapKernel <<<dimGrid, dimBlock >>> (maxWeightD, arraySizeD, valueD, weightD, victorIdD, victorValueD, offsetD, memSizeD, runLength);
    }

    //create host memory
    unsigned long long int *victorIdH = (unsigned long long int*)malloc (sizeof(unsigned long long int) * memSize);
    uint16_t *victorValueH = (uint16_t*)malloc(sizeof(uint16_t) * memSize);

    //initialize to zero
    for (int a1 = 0; a1 < memSize; a1++) {
        victorIdH[a1] = 0;
        victorValueH[a1] = 0;
    }

    cudaMemcpy(victorIdH, victorIdD, sizeof(unsigned long long int) * memSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(victorValueH, victorValueD, sizeof(uint16_t) * memSize, cudaMemcpyDeviceToHost);

    //for debugging what is in memory
    /*for (int a1 = 0; a1 < memSize; a1++) {
        if (victorIdH[a1] != 0) {
            std::cout << a1 << ": " << victorIdH[a1] << ", " << victorValueH[a1] << std::endl;
        }
        
    }*/

    uint16_t highValue = 0;
    unsigned long long int highId = 0;

    for (int a1 = 0; a1 < memSize; a1++) {
        if (victorValueH[a1] > highValue) {
            highValue = victorValueH[a1];
            highId = victorIdH[a1];
        }
    }

    std::cout << "Sets to Search:" << runLength << std::endl;
    std::cout << "Mem Size:" << memSize << std::endl;
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

    free(victorIdH);
    free(victorValueH);


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


