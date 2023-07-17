#pragma once

#include <Eigen/Core>

namespace adso
{
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

// Eigen::Ref typedefs
using MatrixXdRef = Eigen::Ref<Eigen::MatrixXd>;
using VectorXdRef = Eigen::Ref<Eigen::VectorXd>;
using MatrixXdCRef = Eigen::Ref<const Eigen::MatrixXd>;
using VectorXdCRef = Eigen::Ref<const Eigen::VectorXd>;

// Eigen::Map typedefs
using MatrixXdMap = Eigen::Map<Eigen::MatrixXd>;
using VectorXdMap = Eigen::Map<Eigen::VectorXd>;
using VectorXdCMap = Eigen::Map<const Eigen::VectorXd>;
using MatrixXdCMap = Eigen::Map<const Eigen::MatrixXd>;

template <int M, int N>
using MatrixMNd = Eigen::Matrix<double, M, N>;
template <int N>
using VectorNd = Eigen::Matrix<double, N, 1>;
template <int M, int N>
using ArrayMNd = Eigen::Array<double, M, N>;


}