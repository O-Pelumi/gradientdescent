#include "readfile.h"


int readData(const char* filename, Eigen::MatrixXd& data)
{
    const int rows = data.rows();
    const int columns = data.cols();
    int retVal = 1;

    readRows(filename, [rows, columns, &retVal, &data](const size_t row, const std::string line) {
        std::istringstream lineStream(std::move(line));
        std::string val;
        size_t col = 0;

        if (row >= rows) {
            retVal = 0;
            fprintf(stdout, "Read needed rows already\n");
            return retVal;
        }

        while (col < columns && lineStream >> val) {
            char* err;
            data(row, col++) = strtod(val.c_str(), &err);
            if (*err) {
                fprintf(stderr, "Error converting %s to double at index (%zu, %zu)\n", val.c_str(), row, col);
                retVal = -1;
                return retVal;
            }
        }

        if (col != columns) {
            fprintf(stderr, "Empty row or not enough columns in row (%zu)\n", row);
            retVal = -1;
            return retVal;
        }

        return retVal;
    });

    return retVal;
}
