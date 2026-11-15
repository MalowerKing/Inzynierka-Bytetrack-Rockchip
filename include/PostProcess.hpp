#pragma once

#include <string>
#include <vector>
#include "Strack.hpp"
#include <opencv2/core.hpp>

#include "ModelProcessing.hpp"  // OutputTensor, LetterboxInfo
#include <Types.hpp>

// One detection, already mapped back to ORIGINAL CAMERA FRAME pixels.
// This is also the natural input for ByteTrack later.
enum class BoxFormat {
    XYWH_CENTER,  // cx, cy, w, h  (standard YOLOv8 export)
    XYXY          // x1, y1, x2, y2
};

struct PostprocessConfig {
    float     conf_threshold = 0.40f;
    float     nms_threshold  = 0.45f;
    int       max_candidates = 300;   // best candidates kept before NMS
    int       max_detections = 100;
    BoxFormat box_format     = BoxFormat::XYWH_CENTER;
    bool      apply_sigmoid  = false; // set true if class scores turn out to be raw logits
};

// Filled by postprocess() so you can sanity-check the assumptions.
struct PostprocessStats {
    int   num_classes = 0;
    int   num_anchors = 0;
    bool  coords_normalized = false;  // true -> box values were 0..1 and got scaled by the model size
    float max_coord = 0.0f;
    float max_score = 0.0f;           // best class score seen over ALL anchors
    int   candidates  = 0;            // anchors above conf_threshold (before NMS)
    std::vector<float> class_max_score;  // best score of EACH class over all anchors
};

// Expects out.dims == [1, 4 + num_classes, num_anchors] (channel-major), as in your model.
std::vector<Detection> postprocess(const OutputTensor& out,
                                   const LetterboxInfo& lb,
                                   int model_w, int model_h,
                                   int frame_w, int frame_h,
                                   const PostprocessConfig& cfg = PostprocessConfig(),
                                   PostprocessStats* stats = nullptr);

// Draws boxes + "name score" labels onto `frame` (BGR). class_names is optional.
void drawTracks(cv::Mat& frame, const std::vector<STrackPtr>& tracks);
