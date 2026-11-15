#include "PostProcess.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float dequantize(int8_t q, int32_t zp, float scale) {
    return (static_cast<int32_t>(q) - zp) * scale;
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width,  b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float uni   = a.width * a.height + b.width * b.height - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

std::vector<Detection> postprocess(const OutputTensor& out,
                                   const LetterboxInfo& lb,
                                   int model_w, int model_h,
                                   int frame_w, int frame_h,
                                   const PostprocessConfig& cfg,
                                   PostprocessStats* stats) {
    std::vector<Detection> result;
    if (stats) *stats = PostprocessStats();

    // ---- validate layout: [1, 4 + classes, anchors] ----
    if (out.dims.size() != 3 || out.dims[0] != 1 || out.dims[1] <= 4 || out.dims[2] == 0 ||
        lb.scale <= 0.0f || out.scale <= 0.0f) {
        std::cerr << "postprocess: unexpected layout/quantisation (need [1, 4+classes, anchors], scale > 0)\n";
        return result;
    }
    const size_t C = out.dims[1];
    const size_t N = out.dims[2];
    const size_t num_classes = C - 4;
    if (out.data.size() != C * N) {
        std::cerr << "postprocess: data size does not match dims\n";
        return result;
    }

    const int8_t* d   = out.data.data();
    const int32_t zp  = out.zp;
    const float scale = out.scale;

    const int8_t* c0 = d;          // cx (or x1)
    const int8_t* c1 = d + N;      // cy (or y1)
    const int8_t* c2 = d + 2 * N;  // w  (or x2)
    const int8_t* c3 = d + 3 * N;  // h  (or y2)

    // ---- normalised (0..1) or model-pixel coordinates? ----
    // Dequantisation is monotonic, so the max can be taken on raw int8 values.
    const float max_coord =
        std::max(0.0f, dequantize(*std::max_element(d, d + 4 * N), zp, scale));
    const bool  normalized = (max_coord <= 2.0f);
    const float sx = normalized ? static_cast<float>(model_w) : 1.0f;
    const float sy = normalized ? static_cast<float>(model_h) : 1.0f;

    // ---- confidence threshold moved into the raw int8 domain (early reject) ----
    float thr_val = cfg.conf_threshold;
    if (cfg.apply_sigmoid) {
        const float p = std::min(std::max(cfg.conf_threshold, 1e-6f), 1.0f - 1e-6f);
        thr_val = std::log(p / (1.0f - p));   // inverse sigmoid
    }
    // floor() and -1 keep it conservative; the exact float check below is the real filter
    const float raw_f   = std::floor(thr_val / scale) + static_cast<float>(zp) - 1.0f;
    const int   min_raw = static_cast<int>(std::min(std::max(raw_f, -128.0f), 128.0f));

    // ---- decode every anchor above the threshold ----
    std::vector<Detection> cands;

    for (size_t i = 0; i < N; ++i) {
        // best class on raw values
        int8_t best_q = d[4 * N + i];
        size_t best_c = 0;
        for (size_t c = 1; c < num_classes; ++c) {
            const int8_t q = d[(4 + c) * N + i];
            if (q > best_q) { best_q = q; best_c = c; }
        }
        if (static_cast<int>(best_q) < min_raw) continue;

        float best_s = dequantize(best_q, zp, scale);
        if (cfg.apply_sigmoid) best_s = sigmoid(best_s);
        if (best_s < cfg.conf_threshold) continue;

        const float v0 = dequantize(c0[i], zp, scale);
        const float v1 = dequantize(c1[i], zp, scale);
        const float v2 = dequantize(c2[i], zp, scale);
        const float v3 = dequantize(c3[i], zp, scale);

        // box in model-input pixels
        float x1, y1, x2, y2;
        if (cfg.box_format == BoxFormat::XYWH_CENTER) {
            const float cx = v0 * sx, cy = v1 * sy;
            const float bw = v2 * sx, bh = v3 * sy;
            x1 = cx - bw * 0.5f;  y1 = cy - bh * 0.5f;
            x2 = cx + bw * 0.5f;  y2 = cy + bh * 0.5f;
        } else {
            x1 = v0 * sx;  y1 = v1 * sy;
            x2 = v2 * sx;  y2 = v3 * sy;
        }

        // undo letterbox
        x1 = (x1 - lb.pad_x) / lb.scale;
        x2 = (x2 - lb.pad_x) / lb.scale;
        y1 = (y1 - lb.pad_y) / lb.scale;
        y2 = (y2 - lb.pad_y) / lb.scale;

        // clip to frame
        x1 = std::min(std::max(x1, 0.0f), static_cast<float>(frame_w));
        x2 = std::min(std::max(x2, 0.0f), static_cast<float>(frame_w));
        y1 = std::min(std::max(y1, 0.0f), static_cast<float>(frame_h));
        y2 = std::min(std::max(y2, 0.0f), static_cast<float>(frame_h));
        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;

        Detection det;
        det.box      = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        det.score    = best_s;
        det.class_id = static_cast<int>(best_c);
        cands.push_back(det);
    }

    // ---- optional diagnostics (separate pass, only when requested) ----
    if (stats) {
        std::vector<float> class_max(num_classes);
        float max_score = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            const int8_t* row = d + (4 + c) * N;
            float v = dequantize(*std::max_element(row, row + N), zp, scale);
            if (cfg.apply_sigmoid) v = sigmoid(v);
            class_max[c] = v;
            max_score = std::max(max_score, v);
        }
        stats->num_classes       = static_cast<int>(num_classes);
        stats->num_anchors       = static_cast<int>(N);
        stats->coords_normalized = normalized;
        stats->max_coord         = max_coord;
        stats->max_score         = max_score;
        stats->candidates        = static_cast<int>(cands.size());
        stats->class_max_score   = class_max;
    }

    // ---- keep the best max_candidates, then class-aware NMS ----
    auto by_score = [](const Detection& a, const Detection& b) { return a.score > b.score; };
    if (static_cast<int>(cands.size()) > cfg.max_candidates) {
        std::nth_element(cands.begin(), cands.begin() + cfg.max_candidates, cands.end(), by_score);
        cands.resize(cfg.max_candidates);
    }
    std::sort(cands.begin(), cands.end(), by_score);

    std::vector<bool> suppressed(cands.size(), false);
    for (size_t i = 0; i < cands.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(cands[i]);
        if (static_cast<int>(result.size()) >= cfg.max_detections) break;

        for (size_t j = i + 1; j < cands.size(); ++j) {
            if (suppressed[j] || cands[j].class_id != cands[i].class_id) continue;
            if (iou(cands[i].box, cands[j].box) > cfg.nms_threshold) suppressed[j] = true;
        }
    }
    return result;
}

void drawTracks(cv::Mat& frame, const std::vector<STrackPtr>& tracks) {
    static const cv::Scalar palette[] = {
        cv::Scalar(0, 0, 255),   cv::Scalar(0, 200, 0),   cv::Scalar(255, 0, 0),
        cv::Scalar(0, 200, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 200, 0),
        cv::Scalar(0, 128, 255), cv::Scalar(128, 0, 255)
    };
    const int n_colors = static_cast<int>(sizeof(palette) / sizeof(palette[0]));

    for (const STrackPtr& t : tracks) {
        const int id = t->get_track_id();
        const cv::Scalar color = palette[id % n_colors];

        const cv::Rect2f r = t->get_rect();
        const cv::Point p1(cvRound(r.x), cvRound(r.y));
        const cv::Point p2(cvRound(r.x + r.width), cvRound(r.y + r.height));
        cv::rectangle(frame, p1, p2, color, 2);

        const std::string label = "ID " + std::to_string(id);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int ty = std::max(p1.y, ts.height + 4);  // keep the label inside the image

        cv::rectangle(frame, cv::Point(p1.x, ty - ts.height - 4),
                      cv::Point(p1.x + ts.width, ty + baseline - 2), color, -1);
        cv::putText(frame, label, cv::Point(p1.x, ty - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
