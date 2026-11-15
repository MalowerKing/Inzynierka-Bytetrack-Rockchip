#include "KalmanFilter.hpp"

// State (8D):       [cx, cy, a, h, vcx, vcy, va, vh]
// Measurement (4D): [cx, cy, a, h]
//   cx, cy - bounding box center
//   a      - aspect ratio (width / height)
//   h      - height

KalmanFilter::KalmanFilter()
{
    constexpr int   ndim = 4;
    constexpr float dt   = 1.0f; // one frame per step

    // F: constant velocity model
    _motion_mat.setIdentity();
    for (int i = 0; i < ndim; ++i) {
        _motion_mat(i, ndim + i) = dt;
    }

    // H: we observe only position part of the state
    _update_mat.setZero();
    _update_mat.block<4, 4>(0, 0).setIdentity();

    _std_weight_position = 1.0f / 20.0f;
    _std_weight_velocity = 1.0f / 160.0f;
}

void KalmanFilter::initiate(const MeasurementMean& measurement,
                            StateMean& mean,
                            StateCov& covariance)
{
    // Position from the detection, zero initial velocity
    mean.head<4>() = measurement;
    mean.tail<4>().setZero();

    const float h = measurement(3);

    Eigen::Matrix<float, 8, 1> sigma;
    sigma << 2.0f  * _std_weight_position * h,
             2.0f  * _std_weight_position * h,
             1e-2f,
             2.0f  * _std_weight_position * h,
             10.0f * _std_weight_velocity * h,
             10.0f * _std_weight_velocity * h,
             1e-5f,
             10.0f * _std_weight_velocity * h;

    covariance = sigma.array().square().matrix().asDiagonal();
}

void KalmanFilter::predict(StateMean& mean, StateCov& covariance)
{
    const float h = mean(3);

    // Process noise Q, scaled with the current object height
    Eigen::Matrix<float, 8, 1> sigma;
    sigma << _std_weight_position * h,
             _std_weight_position * h,
             1e-2f,
             _std_weight_position * h,
             _std_weight_velocity * h,
             _std_weight_velocity * h,
             1e-5f,
             _std_weight_velocity * h;

    const StateCov motion_cov = sigma.array().square().matrix().asDiagonal();

    // Mean is a row vector, so x' = F x  becomes  m' = m * F^T
    mean = mean * _motion_mat.transpose();

    // P' = F P F^T + Q
    covariance = _motion_mat * covariance * _motion_mat.transpose() + motion_cov;
}

void KalmanFilter::project(const StateMean& mean,
                           const StateCov& covariance,
                           MeasurementMean& proj_mean,
                           MeasurementCov& proj_cov)
{
    const float h = mean(3);

    // Measurement noise R
    Eigen::Matrix<float, 4, 1> sigma;
    sigma << _std_weight_position * h,
             _std_weight_position * h,
             1e-1f,
             _std_weight_position * h;

    const MeasurementCov measurement_cov = sigma.array().square().matrix().asDiagonal();

    // z = H x   ->   row form: m * H^T
    proj_mean = mean * _update_mat.transpose();

    // S = H P H^T + R
    proj_cov = _update_mat * covariance * _update_mat.transpose() + measurement_cov;
}

void KalmanFilter::update(StateMean& mean,
                          StateCov& covariance,
                          const MeasurementMean& measurement)
{
    MeasurementMean proj_mean;
    MeasurementCov  proj_cov;
    project(mean, covariance, proj_mean, proj_cov);

    // Kalman gain K = P H^T S^-1  (8x4).
    // S and P are symmetric, so K^T = S^-1 (H P)  (4x8) - solved without an explicit inverse.
    const Eigen::Matrix<float, 4, 8> kalman_gain_t =
        proj_cov.ldlt().solve(_update_mat * covariance);

    // Innovation y = z - H x  (row vector)
    const MeasurementMean innovation = measurement - proj_mean;

    // x' = x + K y   ->   row form: m + y * K^T
    mean += innovation * kalman_gain_t;

    // P' = P - K S K^T
    covariance -= kalman_gain_t.transpose() * proj_cov * kalman_gain_t;

    // Keep P symmetric (float round-off); eval() avoids aliasing with transpose()
    covariance = (0.5f * (covariance + covariance.transpose())).eval();
}
