#pragma once
#include "Types.hpp"
#include "KalmanFilter.hpp"
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

enum class TrackState {
    New,
    Tracked,
    Lost,
    Removed
};

class STrack {
public:
    STrack(const Detection& detection);
    ~STrack() = default;

    // Główne operacje cyklu życia
    void activate(KalmanFilter& kalman, int frame_id);
    void predict(KalmanFilter& kalman);
    void update(KalmanFilter& kalman, const Detection& new_detection, int frame_id);
    void mark_lost();
    void mark_removed();

    // Gettery
    int get_track_id() const { return track_id; }
    TrackState get_state() const { return state; }
    
    // Zwraca zaktualizowany bounding box (po predykcji Kalmana lub aktualizacji)
    cv::Rect2f get_rect() const; 

public:
    bool is_activated;
    int track_id;
    TrackState state;

    float score;
    int start_frame;
    int tracklet_len;
    int time_since_update; // Licznik klatek od ostatniego dopasowania

    // Stan filtru Kalmana
    KalmanFilter::StateMean mean;
    KalmanFilter::StateCov covariance;

private:
    Detection _current_det; // Zapis pełnej struktury Detection z ostatniego pomiaru
    static int _next_id;
    int next_id();
};

using STrackPtr = std::shared_ptr<STrack>;
