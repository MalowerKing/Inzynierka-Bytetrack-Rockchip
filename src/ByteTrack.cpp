#include "ByteTrack.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>

namespace {

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
constexpr float       kMinScore           = 0.1f;   // detections below this are discarded
constexpr float       kNmsIouThresh       = 0.7f;
constexpr std::size_t kMaxNmsCandidates   = 1000;   // keep NMS cheap
constexpr float       kLowMatchThresh     = 0.5f;   // 2nd association (low-score detections)
constexpr float       kUnconfirmedThresh  = 0.7f;   // association of unconfirmed tracks
constexpr float       kDuplicateDist      = 0.15f;  // tracked/lost duplicate removal
constexpr bool        kFuseScore          = true;   // cost = 1 - IoU * score (as in ByteTrack)

using AssocResult = std::tuple<std::vector<std::pair<int, int>>,
                               std::vector<int>,
                               std::vector<int>>;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
float iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    const float area_a = a.width * a.height;
    const float area_b = b.width * b.height;
    if (area_a <= 0.0f || area_b <= 0.0f) {
        return 0.0f;
    }

    const float iw = std::min(a.x + a.width,  b.x + b.width)  - std::max(a.x, b.x);
    const float ih = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);
    if (iw <= 0.0f || ih <= 0.0f) {
        return 0.0f;
    }

    const float inter = iw * ih;
    const float uni   = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// cost = 1 - IoU * detection_score
void fuse_score(std::vector<std::vector<float>>& cost, const std::vector<STrackPtr>& dets)
{
    for (auto& row : cost) {
        for (std::size_t j = 0; j < row.size(); ++j) {
            row[j] = 1.0f - (1.0f - row[j]) * dets[j]->score;
        }
    }
}

// ---------------------------------------------------------------------------
// Hungarian algorithm (square matrix, row-major, minimisation).
// Returns for every row the assigned column.
// ---------------------------------------------------------------------------
std::vector<int> solve_assignment(const std::vector<double>& cost, int n)
{
    const double INF = std::numeric_limits<double>::max() / 4.0;

    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int>    p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char>   used(n + 1, 0);

        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = INF;
            int j1 = 0;

            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else         { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> row_to_col(n, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] > 0) row_to_col[p[j] - 1] = j - 1;
    }
    return row_to_col;
}

// ---------------------------------------------------------------------------
// Tensor decoding
// Assumed layout (YOLOv8-style, no objectness):
//   [1, 4 + num_classes, N]  or  [1, N, 4 + num_classes]
//   channels 0..3 = cx, cy, w, h in model-input pixels, rest = class scores
// ---------------------------------------------------------------------------
std::vector<Detection> decode_detections(const OutputTensor& t, const LetterboxInfo& lb)
{
    std::vector<Detection> out;

    if (t.dims.size() < 2 || t.data.empty() || t.scale <= 0.0f || lb.scale <= 0.0f) {
        return out;
    }

    const std::size_t a = t.dims[t.dims.size() - 2];
    const std::size_t b = t.dims[t.dims.size() - 1];

    // The smaller dimension is the channel count (4 + classes)
    const bool        channels_first = a < b;
    const std::size_t channels       = channels_first ? a : b;
    const std::size_t anchors        = channels_first ? b : a;

    if (channels <= 4 || t.data.size() < channels * anchors) {
        return out;
    }
    const std::size_t num_classes = channels - 4;

    auto at = [&](std::size_t c, std::size_t i) -> int8_t {
        return channels_first ? t.data[c * anchors + i] : t.data[i * channels + c];
    };
    auto deq = [&](int8_t q) -> float {
        return (static_cast<float>(q) - static_cast<float>(t.zp)) * t.scale;
    };

    const float inv_scale = 1.0f / lb.scale;

    for (std::size_t i = 0; i < anchors; ++i) {
        // Dequantisation is monotonic (scale > 0), so argmax can be done on raw int8
        int8_t      best_q = at(4, i);
        std::size_t best_c = 0;
        for (std::size_t c = 1; c < num_classes; ++c) {
            const int8_t q = at(4 + c, i);
            if (q > best_q) { best_q = q; best_c = c; }
        }

        const float score = deq(best_q);
        if (score < kMinScore) continue;

        const float cx = deq(at(0, i));
        const float cy = deq(at(1, i));
        const float w  = deq(at(2, i));
        const float h  = deq(at(3, i));
        if (w <= 0.0f || h <= 0.0f) continue;

        // Undo letterbox: remove padding, then rescale to the original image
        const float x1 = (cx - 0.5f * w - static_cast<float>(lb.pad_x)) * inv_scale;
        const float y1 = (cy - 0.5f * h - static_cast<float>(lb.pad_y)) * inv_scale;

        Detection d;
        d.box      = cv::Rect2f(x1, y1, w * inv_scale, h * inv_scale);
        d.score    = score;
        d.class_id = static_cast<int>(best_c);
        out.push_back(d);
    }
    return out;
}

// Greedy, class-aware NMS
std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh)
{
    std::sort(dets.begin(), dets.end(),
              [](const Detection& x, const Detection& y) { return x.score > y.score; });
    if (dets.size() > kMaxNmsCandidates) {
        dets.resize(kMaxNmsCandidates);
    }

    std::vector<char>      suppressed(dets.size(), 0);
    std::vector<Detection> keep;

    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);

        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] &&
                dets[j].class_id == dets[i].class_id &&
                iou(dets[i].box, dets[j].box) > iou_thresh) {
                suppressed[j] = 1;
            }
        }
    }
    return keep;
}

} // namespace

// ---------------------------------------------------------------------------
// BYTETracker
// ---------------------------------------------------------------------------
BYTETracker::BYTETracker(int frame_rate, int track_buffer,
                         float track_thresh, float high_thresh, float match_thresh)
    : track_thresh(track_thresh)
    , high_thresh(high_thresh)
    , match_thresh(match_thresh)
    , max_time_lost(static_cast<int>(frame_rate / 30.0f * static_cast<float>(track_buffer)))
    , frame_id(0)
{
}


std::vector<std::vector<float>>
BYTETracker::calc_iou_distance(const std::vector<STrackPtr>& tracks,
                               const std::vector<STrackPtr>& detections)
{
    std::vector<cv::Rect2f> track_rects, det_rects;
    track_rects.reserve(tracks.size());
    det_rects.reserve(detections.size());
    for (const auto& t : tracks)     track_rects.push_back(t->get_rect());
    for (const auto& d : detections) det_rects.push_back(d->get_rect());

    std::vector<std::vector<float>> cost(tracks.size(),
                                         std::vector<float>(detections.size(), 1.0f));
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        for (std::size_t j = 0; j < detections.size(); ++j) {
            cost[i][j] = 1.0f - iou(track_rects[i], det_rects[j]);
        }
    }
    return cost;
}

// Returns {matches (row, col), unmatched rows, unmatched cols}.
// Pairs with cost > thresh are never matched. This reproduces lapjv's cost_limit
// behaviour: the matrix is extended so "leave unmatched" costs thresh/2 per side.
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
BYTETracker::linear_assignment(const std::vector<std::vector<float>>& cost_matrix, float thresh)
{
    const int n_rows = static_cast<int>(cost_matrix.size());
    const int n_cols = n_rows > 0 ? static_cast<int>(cost_matrix[0].size()) : 0;

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_rows, unmatched_cols;

    // Only rows/cols with at least one admissible partner can take part
    std::vector<char> row_ok(n_rows, 0), col_ok(n_cols, 0);
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            if (cost_matrix[i][j] <= thresh) { row_ok[i] = 1; col_ok[j] = 1; }
        }
    }

    std::vector<int> rows, cols;
    for (int i = 0; i < n_rows; ++i) if (row_ok[i]) rows.push_back(i);
    for (int j = 0; j < n_cols; ++j) if (col_ok[j]) cols.push_back(j);

    std::vector<char> row_matched(n_rows, 0), col_matched(n_cols, 0);

    if (!rows.empty()) {
        const int    R    = static_cast<int>(rows.size());
        const int    C    = static_cast<int>(cols.size());
        const int    N    = R + C;
        const double half = 0.5 * static_cast<double>(thresh);

        // [ cost  | half ]
        // [ half  |  0   ]
        std::vector<double> ext(static_cast<std::size_t>(N) * N, 0.0);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                double v;
                if (r < R && c < C)       v = cost_matrix[rows[r]][cols[c]];
                else if (r < R || c < C)  v = half;
                else                      v = 0.0;
                ext[static_cast<std::size_t>(r) * N + c] = v;
            }
        }

        const std::vector<int> assign = solve_assignment(ext, N);
        for (int r = 0; r < R; ++r) {
            const int c = assign[r];
            if (c >= 0 && c < C && cost_matrix[rows[r]][cols[c]] <= thresh) {
                matches.emplace_back(rows[r], cols[c]);
                row_matched[rows[r]] = 1;
                col_matched[cols[c]] = 1;
            }
        }
    }

    for (int i = 0; i < n_rows; ++i) if (!row_matched[i]) unmatched_rows.push_back(i);
    for (int j = 0; j < n_cols; ++j) if (!col_matched[j]) unmatched_cols.push_back(j);

    return std::make_tuple(std::move(matches), std::move(unmatched_rows), std::move(unmatched_cols));
}

std::vector<STrackPtr> BYTETracker::update(const std::vector<Detection>& detections)
{
    ++frame_id;

    // Association helper that also copes with empty inputs
    auto associate = [this](const std::vector<STrackPtr>& tracks,
                            const std::vector<STrackPtr>& dets,
                            float thresh, bool fuse) -> AssocResult
    {
        if (tracks.empty() || dets.empty()) {
            std::vector<int> ut(tracks.size()), ud(dets.size());
            std::iota(ut.begin(), ut.end(), 0);
            std::iota(ud.begin(), ud.end(), 0);
            return std::make_tuple(std::vector<std::pair<int, int>>{}, std::move(ut), std::move(ud));
        }
        auto cost = calc_iou_distance(tracks, dets);
        if (fuse && kFuseScore) {
            fuse_score(cost, dets);
        }
        return linear_assignment(cost, thresh);
    };

    // ---- 1. Split detections into D_high / D_low -------------------------
    // Raw detections are kept next to their STrack wrappers (wrappers are used
    // for IoU, raw ones are passed to STrack::update).
    std::vector<Detection> high_dets, low_dets;
    std::vector<STrackPtr> high_tracks, low_tracks;

    for (const Detection& d : detections) {
        if (d.score >= track_thresh) {
            high_dets.push_back(d);
            high_tracks.push_back(std::make_shared<STrack>(d));
        } else if (d.score > kMinScore) {
            low_dets.push_back(d);
            low_tracks.push_back(std::make_shared<STrack>(d));
        }
    }

    // ---- 2. Predict and associate with high-score detections -------------
    std::vector<STrackPtr> unconfirmed, pool;
    for (const auto& t : tracked_stracks) {
        if (t->is_activated) pool.push_back(t);
        else                 unconfirmed.push_back(t);
    }
    pool.insert(pool.end(), lost_stracks.begin(), lost_stracks.end());

    for (const auto& t : pool) {
        t->predict(kalman_filter);
    }

    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_track, u_high;
    std::tie(matches, u_track, u_high) = associate(pool, high_tracks, match_thresh, true);

    for (const auto& [ti, di] : matches) {
        pool[ti]->update(kalman_filter, high_dets[di], frame_id); // also re-finds Lost tracks
    }

    // ---- 3. Second association: remaining Tracked vs. low-score dets -----
    std::vector<STrackPtr> remaining_tracked;
    for (int ti : u_track) {
        if (pool[ti]->get_state() == TrackState::Tracked) {
            remaining_tracked.push_back(pool[ti]);
        }
    }

    std::tie(matches, u_track, std::ignore) =
        associate(remaining_tracked, low_tracks, kLowMatchThresh, false);

    for (const auto& [ti, di] : matches) {
        remaining_tracked[ti]->update(kalman_filter, low_dets[di], frame_id);
    }
    for (int ti : u_track) {
        remaining_tracked[ti]->mark_lost();
    }

    // ---- 4. Unconfirmed tracks vs. still unmatched high-score dets -------
    std::vector<STrackPtr> remaining_dets;
    std::vector<int>       remaining_idx; // index into high_dets / high_tracks
    for (int di : u_high) {
        remaining_dets.push_back(high_tracks[di]);
        remaining_idx.push_back(di);
    }

    std::vector<int> u_unconfirmed, u_new;
    std::tie(matches, u_unconfirmed, u_new) =
        associate(unconfirmed, remaining_dets, kUnconfirmedThresh, true);

    for (const auto& [ti, ri] : matches) {
        unconfirmed[ti]->update(kalman_filter, high_dets[remaining_idx[ri]], frame_id);
    }
    for (int ti : u_unconfirmed) {
        unconfirmed[ti]->mark_removed();
    }

    // ---- 5. Start new tracks from unmatched, confident detections --------
    std::vector<STrackPtr> born;
    for (int ri : u_new) {
        const STrackPtr& track = high_tracks[remaining_idx[ri]];
        if (track->score < high_thresh) continue;
        track->activate(kalman_filter, frame_id);
        born.push_back(track);
    }

    // ---- 6. Rebuild the track lists from each track's final state --------
    removed_stracks.clear(); // holds only tracks removed during this frame

    std::vector<STrackPtr> new_tracked, new_lost;
    auto sort_track = [&](const STrackPtr& t) {
        switch (t->get_state()) {
        case TrackState::Tracked:
            new_tracked.push_back(t);
            break;
        case TrackState::Lost:
            if (t->time_since_update > max_time_lost) {
                t->mark_removed();
                removed_stracks.push_back(t);
            } else {
                new_lost.push_back(t);
            }
            break;
        case TrackState::Removed:
            removed_stracks.push_back(t);
            break;
        default:
            break;
        }
    };
    for (const auto& t : tracked_stracks) sort_track(t);
    for (const auto& t : lost_stracks)    sort_track(t);
    for (const auto& t : born)            sort_track(t);

    // Remove duplicates between tracked and lost: keep the longer-lived track
    if (!new_tracked.empty() && !new_lost.empty()) {
        const auto dist = calc_iou_distance(new_tracked, new_lost);
        std::vector<char> drop_tracked(new_tracked.size(), 0), drop_lost(new_lost.size(), 0);

        auto age = [this](const STrackPtr& t) {
            return (frame_id - t->time_since_update) - t->start_frame;
        };

        for (std::size_t p = 0; p < new_tracked.size(); ++p) {
            for (std::size_t q = 0; q < new_lost.size(); ++q) {
                if (dist[p][q] < kDuplicateDist) {
                    if (age(new_tracked[p]) > age(new_lost[q])) drop_lost[q]    = 1;
                    else                                        drop_tracked[p] = 1;
                }
            }
        }

        auto apply_drop = [this](std::vector<STrackPtr>& list, const std::vector<char>& drop) {
            std::vector<STrackPtr> kept;
            for (std::size_t i = 0; i < list.size(); ++i) {
                if (drop[i]) {
                    list[i]->mark_removed();
                    removed_stracks.push_back(list[i]);
                } else {
                    kept.push_back(list[i]);
                }
            }
            list.swap(kept);
        };
        apply_drop(new_tracked, drop_tracked);
        apply_drop(new_lost, drop_lost);
    }

    tracked_stracks = std::move(new_tracked);
    lost_stracks    = std::move(new_lost);

    // Output: only confirmed tracks that were matched in this frame
    std::vector<STrackPtr> output;
    for (const auto& t : tracked_stracks) {
        if (t->is_activated) output.push_back(t);
    }
    return output;
}
