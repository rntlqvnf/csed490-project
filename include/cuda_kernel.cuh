
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring> // For strncpy and strlen
#include <chrono>

struct Person {
    long long id;
    char firstName[30];
    char lastName[30];
    char gender[10];
    long long birthday;
    long long creationDate;
    char locationIP[20];
    char browserUsed[20];
    bool finalDst = false;
};

void filterNodes_gpu(std::vector<Person>& nodes, int predicateColumn, const char* predicateValue, std::unordered_set<long long>& srcNodes);

std::vector<long long> bfs_cuda(long long* h_edgesArray, int* h_edgeIndices, int numNodes, int numEdges, const std::unordered_set<long long>& startNodes, int depth);

void filterNodes_gpu_pinned(std::vector<Person>& nodes, int predicateColumn, const char* predicateValue, std::unordered_set<long long>& srcNodes);