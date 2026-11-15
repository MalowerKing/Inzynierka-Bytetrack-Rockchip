#include "Strack.hpp"

#include <algorithm>

// Static ID counter (IDs start at 1, 0 means "not assigned yet")
int STrack::_next_id = 0;

namespace {

// cv::Rect2f (top-left x, y, w, h) -> Kalman measurement [cx, cy, aspect, height]
KalmanFilter::MeasurementMean to_measurement(const cv::Rect2f& box)
{
    const float h = std::max(box.height, 1e-6f); // avoid division by zero

    KalmanFilter::MeasurementMean m;
    m << box.x + box.width * 0.5f,
         box.y + box.height * 0.5f,
         box.width / h,
         box.height;
    return m;
}

} // namespace

STrack::STrack(const Detection& detection)
    : is_activated(false)
    , track_id(0)
    , state(TrackState::New)
    , score(detection.score)
    , start_frame(0)
    , tracklet_len(0)
    , time_since_update(0)
    , _current_det(detection)
{
    // Fixed-size Eigen types are not zero-initialised by default
    mean.setZero();
    covariance.setZero();
}

int STrack::next_id()
{
    return ++_next_id;
}

void STrack::activate(KalmanFilter& kalman, int frame_id)
{
    kalman.initiate(to_measurement(_current_det.box), mean, covariance);

    track_id          = next_id();
    state             = TrackState::Tracked;
    tracklet_len      = 0;
    time_since_update = 0;
    start_frame       = frame_id;

    // Like the original ByteTrack: tracks born in the very first frame are
    // confirmed immediately, all others must be matched once more first.
    is_activated = (frame_id == 1);
}

void STrack::predict(KalmanFilter& kalman)
{
    if (state == TrackState::New) {
        return; // Kalman state not initialised yet
    }

    // Do not let lost tracks grow/shrink: zero the height velocity
    if (state != TrackState::Tracked) {
        mean(7) = 0.0f;
    }

    kalman.predict(mean, covariance);
    ++time_since_update;
}

void STrack::update(KalmanFilter& kalman, const Detection& new_detection, int /*frame_id*/)
{
    // Works both for a normal Tracked match and for re-finding a Lost track
    kalman.update(mean, covariance, to_measurement(new_detection.box));

    _current_det      = new_detection;
    score             = new_detection.score;
    state             = TrackState::Tracked;
    is_activated      = true;
    time_since_update = 0;
    ++tracklet_len;
}

void STrack::mark_lost()
{
    state = TrackState::Lost;
}

void STrack::mark_removed()
{
    state = TrackState::Removed;
}

cv::Rect2f STrack::get_rect() const
{
    // Not activated yet (e.g. a detection wrapped in an STrack): raw detection box
    if (state == TrackState::New) {
        return _current_det.box;
    }

    // Kalman state [cx, cy, a, h] -> top-left x, y, w, h
    const float h = mean(3);
    const float w = mean(2) * h;
    return cv::Rect2f(mean(0) - w * 0.5f, mean(1) - h * 0.5f, w, h);
}

