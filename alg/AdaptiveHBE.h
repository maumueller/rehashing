//
// Created by Kexin Rong on 2018-11-11.
//

#ifndef HBE_ADAPTIVEHBE_H
#define HBE_ADAPTIVEHBE_H

#include <Eigen/Dense>
#include "SketchLSH.h"
#include "BaseLSH.h"
#include "AdaptiveEstimator.h"
#include "kernel.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class AdaptiveHBE : public AdaptiveEstimator {
public:
    vector<BaseLSH> b_levels;
    vector<SketchLSH> s_levels;
    double tau;

    AdaptiveHBE(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps, bool sketch);

protected:
    std::vector<double> evaluateQuery(VectorXd q, int level);

private:
    bool use_sketch;
    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    void buildLevels(shared_ptr<MatrixXd> X, shared_ptr<Kernel> k, double tau, double eps, bool sketch);

};


#endif //HBE_ADAPTIVEHBE_H
