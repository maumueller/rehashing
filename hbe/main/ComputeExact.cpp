/*
 *  Computes the ground truth KDE give dataset, bandwidth and kernel.
 *
 *  The output file contains density of M random points in the dataset.
 *  Each line in the output file contains a random query point's KDE and index (row number in the dataset).
 *
 *  Example usage:
 *      ./hbe conf/shuttle.cfg gaussian
 *
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include "dataUtils.h"
#include "parseConfig.h"
#include "expkernel.h"
#include "gaussiankernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include <chrono>

void checkBandwidth(parseConfig &cfg, char* scope, double h) {
    int N = cfg.getN();
    int dim = cfg.getDim();
    MatrixXd X = dataUtils::readFile(
            cfg.getDataFile(), cfg.ignoreHeader(), N, cfg.getStartCol(), cfg.getEndCol());
    auto band = make_unique<Bandwidth>(N, dim);
    band->useConstant(h);
    shared_ptr<Kernel> kernel;
    if (strcmp(scope, "exp") == 0) {
        kernel = make_shared<Expkernel>(dim);
    } else {
        kernel = make_shared<Gaussiankernel>(dim);
    }
    kernel->initialize(band->bw);
    const double eps = cfg.getEps();
    dataUtils::checkBandwidthSamples(X, eps, kernel);
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char* scope = argv[2];
    parseConfig cfg(argv[1], scope);
    // The dimensionality of each sample vector.
    int d = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = cfg.getM();

    // The bandwidth.
    double h = cfg.getH();
    const char* kernel_type = cfg.getKernel();
    std::cout << "Bandwidth: " << h << std::endl;

    // Read input
    double *x = new double[N * d];
    dataUtils::readFile(cfg.getDataFile(), cfg.ignoreHeader(), N,
            cfg.getStartCol(), cfg.getEndCol(), &x[0]);
    // Check bandwidth
    // checkBandwidth(cfg, scope, h);

    int hasQuery = strcmp(cfg.getDataFile(), cfg.getQueryFile());

    double *y;
    if (hasQuery != 0) {
        y = new double[M * d];
        dataUtils::readFile(cfg.getQueryFile(), cfg.ignoreHeader(), M,
                            cfg.getStartCol(), cfg.getEndCol(), &y[0]);
    }

    double *g = new double[M];

    // Random init
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937_64 rng = std::mt19937_64(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::ofstream outfile(cfg.getExactPath());
    double hSquare = h * h;
    for(int j = 0; j < M; j++) {
        int idx = j;
        if (M < N && hasQuery == 0) {
            idx = distribution(rng);
        }
        g[j] = 0.0;
        for(int i = 0; i < N; i++) {
            double norm = 0.0;
            for (int k = 0; k < d; k++) {
                double temp;
                if (hasQuery != 0) {
                    temp = x[(d*i) + k] - y[(d*idx) + k];
                } else {
                    temp = x[(d*i) + k] - x[(d*idx) + k];
                }
                norm = norm + (temp*temp);
            }
            if (strcmp(kernel_type, "exp") == 0) { // exp kernel
                g[j] = g[j] + exp(-sqrt(norm/hSquare));
            } else {
                g[j] = g[j] + exp(-norm/hSquare);
            }
        }
        g[j] /= N;
        outfile << g[j] << "," << idx << "\n";
    }
    outfile.close();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << M << " queries: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000.0
        << " sec" << std::endl;

}