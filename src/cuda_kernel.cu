// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/cuda_kernel.cuh"

__device__ bool compareStrings(const char* str1, const char* str2) {
    int idx = 0;
    while (str1[idx] != '\0' && str2[idx] != '\0') {
        if (str1[idx] != str2[idx]) {
            return false;
        }
        idx++;
    }
    return str1[idx] == str2[idx];
}


__global__ void filterNodesKernel(Person* nodes, int numNodes, int predicateColumn, const char* predicateValue, bool* resultBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numNodes) {
        Person node = nodes[idx];
        bool result = false;
        result = compareStrings(node.gender, predicateValue);
        resultBuffer[idx] = result;
    }
}

void filterNodes_gpu(std::vector<Person>& nodes, int predicateColumn, const char* predicateValue, std::unordered_set<long long>& srcNodes) {
    Person* deviceNodes;
    bool* deviceResults;
    char* devicePredicateValue;

    // Allocate memory on GPU
    auto startMemory = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&deviceNodes, nodes.size() * sizeof(Person));
    cudaMalloc((void**)&deviceResults, nodes.size() * sizeof(bool));
    cudaMalloc((void**)&devicePredicateValue, strlen(predicateValue) + 1); // +1 for null terminator

    // Copy data to GPU
    cudaMemcpy(deviceNodes, nodes.data(), nodes.size() * sizeof(Person), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePredicateValue, predicateValue, strlen(predicateValue) + 1, cudaMemcpyHostToDevice);
    auto endMemory = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedMemory = endMemory - startMemory;
    std::cout << "Time taken for memory: " << elapsedMemory.count() << " milliseconds" << std::endl;

    // Calculate grid and block sizes
    int blockSize = 256; // Example block size, may need tuning
    int numBlocks = (nodes.size() + blockSize - 1) / blockSize;

    // Launch the kernel
    auto startKernel = std::chrono::high_resolution_clock::now();
    filterNodesKernel<<<numBlocks, blockSize>>>(deviceNodes, nodes.size(), predicateColumn, devicePredicateValue, deviceResults);
    auto endKernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedKernel = endKernel - startKernel;
    std::cout << "Time taken for kernel call: " << elapsedKernel.count() << " milliseconds" << std::endl;

    // Allocate host buffer for results
    bool* hostResults;
    cudaHostAlloc((void**)&hostResults, nodes.size() * sizeof(bool), cudaHostAllocDefault);
    // Copy results back to CPU
    cudaMemcpy(hostResults, deviceResults, nodes.size() * sizeof(bool), cudaMemcpyDeviceToHost);

    // Populate srcNodes based on results
    srcNodes.clear();
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (hostResults[i]) {
            srcNodes.insert(nodes[i].id);
        }
    }

    // Free host and GPU memory
    cudaFreeHost(hostResults);
    cudaFree(deviceNodes);
    cudaFree(deviceResults);
    cudaFree(devicePredicateValue); // Free the allocated memory for predicate value
}

void filterNodes_gpu_pinned(std::vector<Person>& nodes, int predicateColumn, const char* predicateValue, std::unordered_set<long long>& srcNodes) {
    Person* deviceNodes;
    bool* deviceResults;
    char* devicePredicateValue;

    // Allocate pinned memory on host
    Person* hostNodes;
    cudaHostAlloc((void**)&hostNodes, nodes.size() * sizeof(Person), cudaHostAllocDefault);
    memcpy(hostNodes, nodes.data(), nodes.size() * sizeof(Person)); // Copy data to pinned memory

    // Allocate memory on GPU
    auto startMemory = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&deviceNodes, nodes.size() * sizeof(Person));
    cudaMalloc((void**)&deviceResults, nodes.size() * sizeof(bool));
    cudaMalloc((void**)&devicePredicateValue, strlen(predicateValue) + 1);

    // Copy data to GPU using pinned memory
    cudaMemcpyAsync(deviceNodes, hostNodes, nodes.size() * sizeof(Person), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(devicePredicateValue, predicateValue, strlen(predicateValue) + 1, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); // Synchronize to ensure copying is done
    auto endMemory = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedMemory = endMemory - startMemory;
    std::cout << "Time taken for memory: " << elapsedMemory.count() << " milliseconds" << std::endl;

    // Calculate grid and block sizes
    int blockSize = 256; // Example block size, may need tuning
    int numBlocks = (nodes.size() + blockSize - 1) / blockSize;

    // Launch the kernel
    auto startKernel = std::chrono::high_resolution_clock::now();
    filterNodesKernel<<<numBlocks, blockSize>>>(deviceNodes, nodes.size(), predicateColumn, devicePredicateValue, deviceResults);
    cudaDeviceSynchronize(); // Synchronize after kernel execution
    auto endKernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedKernel = endKernel - startKernel;
    std::cout << "Time taken for kernel call: " << elapsedKernel.count() << " milliseconds" << std::endl;

    // Allocate pinned memory for results
    bool* hostResults;
    cudaHostAlloc((void**)&hostResults, nodes.size() * sizeof(bool), cudaHostAllocDefault);

    // Copy results back to CPU using pinned memory
    cudaMemcpyAsync(hostResults, deviceResults, nodes.size() * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Synchronize to ensure copying is done

    // Populate srcNodes based on results
    srcNodes.clear();
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (hostResults[i]) {
            srcNodes.insert(nodes[i].id);
        }
    }

    // Free host and GPU memory
    cudaFreeHost(hostResults);
    cudaFreeHost(hostNodes);
    cudaFree(deviceNodes);
    cudaFree(deviceResults);
    cudaFree(devicePredicateValue);
}

__global__ void bfsKernel(long long* edgesArray, int* edgeIndices, bool* visited, int* depthArray, int maxDepth, bool* frontier, bool* newFrontier, int numNodes) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < numNodes && frontier[threadId] && !visited[threadId]) {
        visited[threadId] = true;
        int startEdge = edgeIndices[threadId];
        int endEdge = edgeIndices[threadId + 1];

        for (int edge = startEdge; edge < endEdge; ++edge) {
            int neighbor = edgesArray[edge];
            if (!visited[neighbor] && depthArray[threadId] + 1 <= maxDepth) {
                depthArray[neighbor] = depthArray[threadId] + 1;
                newFrontier[neighbor] = true;
            }
        }
    }
}

__global__ void bfsKernelOptimized(long long* edgesArray, int* edgeIndices, bool* visited, int* depthArray, int maxDepth, bool* frontier, bool* newFrontier, int numNodes) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= numNodes) return;

    __shared__ bool sharedFrontier[256]; // Assuming blockDim.x is 256
    if (threadIdx.x < numNodes) {
        sharedFrontier[threadIdx.x] = frontier[threadId];
    }
    __syncthreads();

    if (sharedFrontier[threadIdx.x] && !visited[threadId]) {
        visited[threadId] = true;
        int startEdge = edgeIndices[threadId];
        int endEdge = edgeIndices[threadId + 1];

        for (int edge = startEdge; edge < endEdge; ++edge) {
            int neighbor = edgesArray[edge];
            if (!visited[neighbor] && depthArray[threadId] + 1 <= maxDepth) {
                depthArray[neighbor] = depthArray[threadId] + 1;
                newFrontier[neighbor] = true;
            }
        }
    }
}


__global__ void processEdges(long long* edgesArray, int startEdge, int endEdge, int parent, int* depthArray, int maxDepth, bool* newFrontier, bool* visited) {
    int edgeId = blockIdx.x * blockDim.x + threadIdx.x + startEdge;
    if (edgeId < endEdge) {
        int neighbor = edgesArray[edgeId];
        if (!visited[neighbor] && depthArray[parent] + 1 <= maxDepth) {
            depthArray[neighbor] = depthArray[parent] + 1;
            newFrontier[neighbor] = true;
        }
    }
}

__global__ void bfsKernelDP(long long* edgesArray, int* edgeIndices, bool* visited, int* depthArray, int maxDepth, bool* frontier, bool* newFrontier, int numNodes) {
    __shared__ bool sharedFrontier[256]; // Adjust size according to your threads per block
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load frontier into shared memory
    if (threadIdx.x < numNodes) {
        sharedFrontier[threadIdx.x] = frontier[threadId];
    }
    __syncthreads();

    if (threadId < numNodes && sharedFrontier[threadIdx.x] && !visited[threadId]) {
        visited[threadId] = true;
        int startEdge = edgeIndices[threadId];
        int endEdge = edgeIndices[threadId + 1];

        if (endEdge - startEdge > 0) {
            int threadsPerBlock = 256;
            int blocksPerGrid = (endEdge - startEdge + threadsPerBlock - 1) / threadsPerBlock;
            processEdges<<<blocksPerGrid, threadsPerBlock>>>(edgesArray, startEdge, endEdge, threadId, depthArray, maxDepth, newFrontier, visited);
        }
    }
}

std::vector<long long> bfs_cuda(long long* h_edgesArray, int* h_edgeIndices, int numNodes, int numEdges, const std::unordered_set<long long>& startNodes, int depth) {
    // Device variables
    long long* d_edgesArray;
    int* d_edgeIndices;
    bool* d_visited, *d_frontier, *d_newFrontier;
    int* d_depthArray;

    // Allocate device memory
    cudaMalloc(&d_edgesArray, sizeof(long long) * numEdges);
    cudaMalloc(&d_edgeIndices, sizeof(int) * (numNodes + 1));
    cudaMalloc(&d_visited, sizeof(bool) * numNodes);
    cudaMalloc(&d_frontier, sizeof(bool) * numNodes);
    cudaMalloc(&d_newFrontier, sizeof(bool) * numNodes);
    cudaMalloc(&d_depthArray, sizeof(int) * numNodes);

    // Copy data from host to device
    cudaMemcpy(d_edgesArray, h_edgesArray, sizeof(long long) * numEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeIndices, h_edgeIndices, sizeof(int) * (numNodes + 1), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, sizeof(bool) * numNodes);
    cudaMemset(d_frontier, 0, sizeof(bool) * numNodes);
    cudaMemset(d_newFrontier, 0, sizeof(bool) * numNodes);
    cudaMemset(d_depthArray, 0, sizeof(int) * numNodes);

    // Initialize the frontier with the start nodes
    for (long long startNode : startNodes) {
        cudaMemcpy(&d_frontier[startNode], &startNode, sizeof(bool), cudaMemcpyHostToDevice);
    }

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

    // BFS iterations
    auto startKernel = std::chrono::high_resolution_clock::now();
    for (int currentDepth = 0; currentDepth <= depth; ++currentDepth) {
        bfsKernelOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_edgesArray, d_edgeIndices, d_visited, d_depthArray, depth, d_frontier, d_newFrontier, numNodes);
        cudaDeviceSynchronize();

        // Swap frontiers
        bool* temp = d_frontier;
        d_frontier = d_newFrontier;
        d_newFrontier = temp;
        cudaMemset(d_newFrontier, 0, sizeof(bool) * numNodes);
    }
    auto endKernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedKernel = endKernel - startKernel;
    std::cout << "Time taken for kernel call: " << elapsedKernel.count() << " milliseconds" << std::endl;

    // Copy results back to host
    bool* h_visited = new bool[numNodes];
    cudaMemcpy(h_visited, d_visited, sizeof(bool) * numNodes, cudaMemcpyDeviceToHost);

    // Cleanup device memory
    cudaFree(d_edgesArray);
    cudaFree(d_edgeIndices);
    cudaFree(d_visited);
    cudaFree(d_frontier);
    cudaFree(d_newFrontier);
    cudaFree(d_depthArray);

    // Convert the result to the desired format
    std::vector<long long> result;
    for (int i = 0; i < numNodes; ++i) {
        if (h_visited[i]) {
            result.push_back(i);
        }
    }
    delete[] h_visited;

    return result;
}
