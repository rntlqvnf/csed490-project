#include "include/cuda_kernel.cuh"
#define CUDA_ENABLED

// Function to safely copy a std::string to a char array
void safeStrCopy(char* dest, const std::string& src, size_t destSize) {
    strncpy(dest, src.c_str(), destSize - 1);
    dest[destSize - 1] = '\0'; // Ensure null termination
}

// Read CSV file and parse it into a vector of Person
std::vector<Person> readNodeCsv(const std::string& filename) {
    std::vector<Person> nodes;
    std::ifstream file(filename);
    std::string line;

    // Skip the header
    getline(file, line);

    while (getline(file, line)) {
        std::istringstream s(line);
        Person person;
        std::string field;

        // Parse the id
        getline(s, field, '|');
        person.id = std::stoll(field);

        // Parse the firstName
        getline(s, field, '|');
        safeStrCopy(person.firstName, field, sizeof(person.firstName));

        // Parse the lastName
        getline(s, field, '|');
        safeStrCopy(person.lastName, field, sizeof(person.lastName));

        // Parse the gender
        getline(s, field, '|');
        safeStrCopy(person.gender, field, sizeof(person.gender));

        // Parse the birthday
        getline(s, field, '|');
        person.birthday = std::stoll(field);

        // Parse the creationDate
        getline(s, field, '|');
        person.creationDate = std::stoll(field);

        // Parse the locationIP
        getline(s, field, '|');
        safeStrCopy(person.locationIP, field, sizeof(person.locationIP));

        // Parse the browserUsed
        getline(s, field, '|');
        safeStrCopy(person.browserUsed, field, sizeof(person.browserUsed));

        nodes.push_back(person);
    }

    return nodes;
}


// Read edge CSV and parse it into a map of connections
std::unordered_map<long long, std::vector<long long>> readEdgeCsv(const std::string& filename) {
    std::unordered_map<long long, std::vector<long long>> edges;
    std::ifstream file(filename);
    std::string line, field;
    long long startId, endId;
    
    // Skip the header
    getline(file, line);

    while (getline(file, line)) {
        std::stringstream s(line);
        getline(s, field, '|');
        startId = std::stoll(field);
        getline(s, field, '|');
        endId = std::stoll(field);
        edges[startId].push_back(endId);
    }

    return edges;
}

// BFS with depth limitation
std::vector<long long> bfs(const std::unordered_map<long long, std::vector<long long>>& edges, 
                           const std::unordered_set<long long>& startNodes, 
                           int depth) {
    std::vector<long long> result;
    std::queue<std::pair<long long, int>> q;
    std::unordered_set<long long> visited;

    for (const auto& node : startNodes) {
        q.push({node, 0});
        visited.insert(node);
    }

    while (!q.empty()) {
        auto [current, currentDepth] = q.front();
        q.pop();

        // Add nodes that are at or under the specified depth
        if (currentDepth <= depth) {
            result.push_back(current);
        }

        // Continue exploring neighbors if the current depth is less than the maximum depth
        if (currentDepth < depth) {
            // Check if the current node has neighbors before accessing them
            if (edges.find(current) != edges.end()) {
                for (const auto& neighbor : edges.at(current)) {
                    if (visited.find(neighbor) == visited.end()) {
                        q.push({neighbor, currentDepth + 1});
                        visited.insert(neighbor);
                    }
                }
            }
        }
    }

    return result;
}

// Predicate function for filtering nodes
bool predicate(const Person& person, int column, const std::string& value) {
    switch (column) {
        case 0: return person.id == std::stoll(value);
        case 1: return person.firstName == value;
        case 2: return person.lastName == value;
        case 3: return person.gender == value;
        case 4: return person.birthday == std::stoll(value);
        case 5: return person.creationDate == std::stoll(value);
        case 6: return person.locationIP == value;
        case 7: return person.browserUsed == value;
        default: return true;
    }
}

void filterSrcNodes(std::unordered_set<long long>& srcNodes, std::vector<Person>& nodes, int srcPredicateColumn, std::string& srcPredicateValue) {
    for (const auto& node : nodes) {
        if (predicate(node, srcPredicateColumn, srcPredicateValue)) {
            srcNodes.insert(node.id);
        }
    }
}

void filterDstNodes(std::vector<long long>& dstNodes, std::vector<Person>& nodes, int dstPredicateColumn, std::string& dstPredicateValue) {
    for (const auto& nodeId : dstNodes) {
        Person& p = nodes[nodeId];
        if (predicate(p, dstPredicateColumn, dstPredicateValue)) {
            p.finalDst = true;
        }
    }
}

void mapNodeIDs(std::vector<Person>& nodes, std::unordered_map<long long, long long>& idMapping) {
    long long physicalId = 0;
    for (auto& node : nodes) {
        idMapping[node.id] = physicalId;
        node.id = physicalId++;
    }
}

void mapEdges(std::unordered_map<long long, std::vector<long long>>& edges, std::unordered_map<long long, long long>& idMapping, std::unordered_map<long long, std::vector<long long>>& originalEdges) {
    for (const auto& [startId, endIds] : originalEdges) {
        int physicalStartId = idMapping[startId];
        std::vector<long long> physicalEndIds;
        for (auto endId : endIds) {
            physicalEndIds.push_back(idMapping[endId]);
        }
        edges[physicalStartId] = physicalEndIds;
    }
}

void convertGraphToCudaFormat(const std::unordered_map<long long, std::vector<long long>>& edgesMap, 
                              std::vector<long long>& edgesArray, 
                              std::vector<int>& edgeIndices) {
    int totalEdges = 0;
    for (const auto& pair : edgesMap) {
        totalEdges += pair.second.size();
    }

    edgesArray.resize(totalEdges);
    edgeIndices.resize(edgesMap.size() + 1);

    int currentIndex = 0;
    int nodeIndex = 0;
    for (const auto& pair : edgesMap) {
        edgeIndices[nodeIndex++] = currentIndex;
        for (long long neighbor : pair.second) {
            edgesArray[currentIndex++] = neighbor;
        }
    }
    edgeIndices[nodeIndex] = currentIndex; // Mark the end of the last adjacency list
}

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <Node.csv> <Edge.csv> <Depth> <SrcPredicateColumn> <SrcPredicateValue> <DstPredicateColumn> <DstPredicateValue>" << std::endl;
        return 1;
    }

    std::string nodeFile = argv[1];
    std::string edgeFile = argv[2];
    int depth = std::stoi(argv[3]);
    int srcPredicateColumn = std::stoi(argv[4]);
    std::string srcPredicateValue = argv[5];
    int dstPredicateColumn = std::stoi(argv[6]);
    std::string dstPredicateValue = argv[7];

    auto nodes = readNodeCsv(nodeFile);
    auto originalEdges = readEdgeCsv(edgeFile);

    // Create mapping from original ID to physical ID
    std::unordered_map<long long, long long> idMapping;
    mapNodeIDs(nodes, idMapping);

    // Convert original edges to use physical IDs
    std::unordered_map<long long, std::vector<long long>> edges;
    mapEdges(edges, idMapping, originalEdges);

    // Convert the graph to a CUDA-friendly format
    std::vector<long long> edgesArray;
    std::vector<int> edgeIndices;
    convertGraphToCudaFormat(edges, edgesArray, edgeIndices);

    // Filter source nodes
    auto startFilterSrc = std::chrono::high_resolution_clock::now();
    std::unordered_set<long long> srcNodes;
#ifdef CUDA_ENABLED
    filterNodes_gpu_pinned(nodes, srcPredicateColumn, srcPredicateValue.c_str(), srcNodes);
#else
    filterSrcNodes(srcNodes, nodes, srcPredicateColumn, srcPredicateValue);
#endif
    auto endFilterSrc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedFilterSrc = endFilterSrc - startFilterSrc;
    std::cout << "Time taken for filtering source nodes: " << elapsedFilterSrc.count() << " milliseconds" << std::endl;

    // Perform BFS
    auto startBfs = std::chrono::high_resolution_clock::now();
#ifdef CUDA_ENABLED
    auto dstNodes = bfs_cuda(edgesArray.data(), edgeIndices.data(), nodes.size(), edgesArray.size(), srcNodes, depth);
#else
    auto dstNodes = bfs(edges, srcNodes, depth);
#endif
    auto endBfs = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedBfs = endBfs - startBfs;
    std::cout << "Time taken for BFS: " << elapsedBfs.count() << " milliseconds" << std::endl;

    // Filter destination nodes and print
    auto startFilterDst = std::chrono::high_resolution_clock::now();
    filterDstNodes(dstNodes, nodes, dstPredicateColumn, dstPredicateValue);
    auto endFilterDst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedFilterDst = endFilterDst - startFilterDst;
    std::cout << "Time taken for filtering destination nodes: " << elapsedFilterDst.count() << " milliseconds" << std::endl;

    return 0;
}
