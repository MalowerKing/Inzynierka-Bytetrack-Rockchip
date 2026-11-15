#pragma once
#include <Eigen/Dense>

class KalmanFilter {
public:
    KalmanFilter();
    ~KalmanFilter() = default;

    // Typy dla macierzy stanu (8D) i pomiaru (4D)
    using StateMean = Eigen::Matrix<float, 1, 8>;
    using StateCov = Eigen::Matrix<float, 8, 8>;
    using MeasurementMean = Eigen::Matrix<float, 1, 4>;
    using MeasurementCov = Eigen::Matrix<float, 4, 4>;

    // Inicjalizuje stan na podstawie pierwszej detekcji
    void initiate(const MeasurementMean& measurement, StateMean& mean, StateCov& covariance);

    // Przewiduje pozycję w kolejnej klatce (Linia 14-16)
    void predict(StateMean& mean, StateCov& covariance);

    // Aktualizuje stan na podstawie dopasowanej detekcji
    void update(StateMean& mean, StateCov& covariance, const MeasurementMean& measurement);

    // Mapuje stan 8D na przestrzeń pomiaru 4D do obliczeń odległości
    void project(const StateMean& mean, const StateCov& covariance, MeasurementMean& proj_mean, MeasurementCov& proj_cov);

private:
    Eigen::Matrix<float, 8, 8> _motion_mat;
    Eigen::Matrix<float, 4, 8> _update_mat;
    float _std_weight_position;
    float _std_weight_velocity;
};
