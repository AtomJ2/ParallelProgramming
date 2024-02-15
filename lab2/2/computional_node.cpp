#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


std::string executeCommand(const std::string& command) {
    std::string result = "";
    char buffer[128];
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe) {
        while (!feof(pipe))
            if (fgets(buffer, 128, pipe) != nullptr)
                result += buffer;
        pclose(pipe);
    }
    return result;
}

int main() {
    std::string cpuInfo = executeCommand("lscpu");

    std::string serverName = executeCommand("cat /sys/devices/virtual/dmi/id/product_name");

    std::string numaInfo = executeCommand("numactl --hardware");

    std::istringstream numaStream(numaInfo);
    std::string numaLine;
    int numaNodeCount = 0;

    while (std::getline(numaStream, numaLine))
        if (numaLine.find("node") != std::string::npos)
            numaNodeCount++;

    std::string osInfo = executeCommand("cat /etc/os-release");

    std::cout << "CPU info:\n" << cpuInfo << std::endl;
    std::cout << "Server: " << serverName << std::endl;
    std::cout << "Number of NUMA Nodes: " << numaNodeCount << std::endl;
    std::cout << "OS info:\n" << osInfo << std::endl;

    return 0;
}