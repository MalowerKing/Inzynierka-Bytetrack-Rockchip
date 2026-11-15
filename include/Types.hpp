#pragma once
#include <vector>
#include <cstdint>

#include <opencv2/core.hpp>

// Surowy tensor wejściowy z modelu
struct OutputTensor {
    std::vector<uint32_t> dims;
    std::vector<int8_t>   data;
    float scale = 1.0f;
    int32_t zp = 0;
};


struct Detection {
    cv::Rect2f box;        // x, y, width, height
    float      score    = 0.0f;
    int        class_id = -1;
};

struct LetterboxInfo {
    float scale = 1.0f;
    int   pad_x = 0;
    int   pad_y = 0;
};
