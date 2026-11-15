#pragma once
#include "Types.hpp"
#include "Strack.hpp"
#include "KalmanFilter.hpp"
#include <vector>
#include <tuple>
#include <opencv2/core.hpp>

class BYTETracker {
public:
    // track_thresh = tau (splits D_high / D_low)
    BYTETracker(int frame_rate = 30, int track_buffer = 30, float track_thresh = 0.5f,
                float high_thresh = 0.6f, float match_thresh = 0.8f);
    ~BYTETracker() = default;

    // Input: output of postprocess() (original-frame pixels, already NMS-ed and clipped)
    std::vector<STrackPtr> update(const std::vector<Detection>& detections);

private:
    float track_thresh; // tau
    float high_thresh;
    float match_thresh;
    int max_time_lost;
    int frame_id;

    KalmanFilter kalman_filter;

    std::vector<STrackPtr> tracked_stracks;
    std::vector<STrackPtr> lost_stracks;
    std::vector<STrackPtr> removed_stracks;

    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    linear_assignment(const std::vector<std::vector<float>>& cost_matrix, float thresh);

    std::vector<std::vector<float>> calc_iou_distance(const std::vector<STrackPtr>& tracks,
                                                      const std::vector<STrackPtr>& detections);
};;
