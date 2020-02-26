#ifndef READFILE_HPP
#define READFILE_HPP

#include <fstream>
#include <Eigen/Dense>


template<typename fn>
int readRows(const char* filename, fn rowHandler)
{
    std::ifstream data(filename, std::ios_base::in);
    std::string line;
    size_t row = 0;

    if (!data) {
        printf("Unable to open file: %s", filename);
        return -1;
    }

    while (std::getline(data, line)) {
        if (rowHandler(row++, std::move(line)) <= 0) {
            break;
        }
    }

    return row;
}

int readData(const char* filename, Eigen::MatrixXd& data);

#endif
