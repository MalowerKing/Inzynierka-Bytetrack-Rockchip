#include "ModelProcessing.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <rknn_api.h>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/imgproc.hpp> 


namespace {
 
std::string dimsToString(const rknn_tensor_attr& a) {
    std::string s = "[";
    for (uint32_t i = 0; i < a.n_dims; ++i) {
        if (i) s += ", ";
        s += std::to_string(a.dims[i]);
    }
    return s + "]";
}
 
const char* typeName(rknn_tensor_type t) {
    switch (t) {
        case RKNN_TENSOR_FLOAT32: return "FLOAT32";
        case RKNN_TENSOR_FLOAT16: return "FLOAT16";
        case RKNN_TENSOR_INT8:    return "INT8";
        case RKNN_TENSOR_UINT8:   return "UINT8";
        default:                  return "OTHER";
    }
}
 
const char* fmtName(rknn_tensor_format f) {
    switch (f) {
        case RKNN_TENSOR_NCHW:     return "NCHW";
        case RKNN_TENSOR_NHWC:     return "NHWC";
        case RKNN_TENSOR_NC1HWC2:  return "NC1HWC2";
        default:                   return "OTHER";
    }
}


void printAttr(const char* tag, const rknn_tensor_attr& a) {
    std::cout << tag << " " << a.index << ": dims=" << dimsToString(a)
              << " fmt=" << fmtName(a.fmt) << " type=" << typeName(a.type)
              << " n_elems=" << a.n_elems << " size=" << a.size
              << " size_with_stride=" << a.size_with_stride
              << " w_stride=" << a.w_stride
              << " zp=" << a.zp << " scale=" << a.scale << "\n";
}

}


bool Model::init(const std::string& model_path) {
        //----------------------------------------------
        //Wgranie modelu do pamieci
        std::vector<uint8_t> model_data = readFileData(model_path);
        if (model_data.empty()) {
                std::cerr << "Failed to read model file.\n";
            return false;
        }
        //----------------------------------------------------------
        //Incjalizacja kontekstu
       int ret = rknn_init(&ctx, model_data.data(), model_data.size(), 0, nullptr); 
        if (ret < 0) {
            std::cerr << "rknn_init error ret=" << ret << "\n";
            return false;
        }
        //--------------------------------------------------------------
        //Rozponanie tensorów wejscia i wejscia
        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if(ret<0) return false;

        //wejscia
        input_attrs.resize(io_num.n_input);
        for (uint32_t i = 0; i < io_num.n_input; ++i) {
            input_attrs[i].index = i;
            rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        }

        //wyjscia
        output_attrs.resize(io_num.n_output);
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            output_attrs[i].index = i;
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        }
       //---------------------------------------------------------
        //Cache wejscia i wyjscia

        
         const rknn_tensor_attr& in = input_attrs[0];

        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            model_channel = in.dims[1];
            model_height  = in.dims[2];

        if (in.n_dims != 4) {
                std::cerr << "Unexpected input rank: " << in.n_dims << "\n";
        return false;
        }
            model_width   = in.dims[3];
        } else { // NHWC
            model_height  = in.dims[1];
            model_width   = in.dims[2];
            model_channel = in.dims[3];
        }

        
    if (model_channel != 3) {
        std::cerr << "Only 3-channel image models are supported, got " << model_channel << "\n";
        return false;
    }
 
    std::cout << "Model input : " << dimsToString(in) << " (" << model_width << "x"
              << model_height << "x" << model_channel << ")\n";
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        std::cout << "Model output " << i << ": " << dimsToString(output_attrs[i]) << "\n";
    }

        
    printAttr("Model input ", input_attrs[0]);
    for (uint32_t i = 0; i < io_num.n_output; ++i) printAttr("Model output", output_attrs[i]);
 
    return setupIoMemory();

}


bool Model::setupIoMemory() {
    // INPUT: we hand the NPU uint8 NHWC (exactly what OpenCV gives us after
    // BGR->RGB). Normalization/quantization is fused into the NPU by the runtime.
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt  = RKNN_TENSOR_NHWC;
 
    const uint32_t in_size = std::max(input_attrs[0].size_with_stride, input_attrs[0].size);
    input_mem = rknn_create_mem(ctx, in_size);
    if (!input_mem) {
        std::cerr << "rknn_create_mem (input) failed\n";
        return false;
    }
    int ret = rknn_set_io_mem(ctx, input_mem, &input_attrs[0]);
    if (ret < 0) {
        std::cerr << "rknn_set_io_mem (input) failed ret=" << ret << "\n";
        return false;
    }
 
    // OUTPUTS: keep the model's own type (int8 on RV1106) and dequantize on the CPU.
    output_mems.assign(io_num.n_output, nullptr);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        const uint32_t sz = std::max(output_attrs[i].size_with_stride, output_attrs[i].size);
        output_mems[i] = rknn_create_mem(ctx, sz);
        if (!output_mems[i]) {
            std::cerr << "rknn_create_mem (output " << i << ") failed\n";
            return false;
        }
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            std::cerr << "rknn_set_io_mem (output " << i << ") failed ret=" << ret << "\n";
            return false;
        }
    }
    return true;
}


bool Model::preprocess(const cv::Mat& bgr, cv::Mat& rgb_out, LetterboxInfo& lb) const {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        std::cerr << "preprocess: expected a non-empty CV_8UC3 (BGR) frame\n";
        return false;
    }
 
    // Skalowanie wejscia
    const float r = std::min(static_cast<float>(model_width)  / bgr.cols,
                             static_cast<float>(model_height) / bgr.rows);
    const int new_w = std::max(1, static_cast<int>(std::lround(bgr.cols * r)));
    const int new_h = std::max(1, static_cast<int>(std::lround(bgr.rows * r)));
 
    lb.scale = r;
    lb.pad_x = (model_width  - new_w) / 2;
    lb.pad_y = (model_height - new_h) / 2;
 
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h));
 
    // Pad up to exactly model_width x model_height (114 = standard YOLO grey)
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       lb.pad_y, model_height - new_h - lb.pad_y,
                       lb.pad_x, model_width  - new_w - lb.pad_x,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
 
    // Konwert BGR -> RBG
    cv::cvtColor(padded, rgb_out, cv::COLOR_BGR2RGB);
 
    if (!rgb_out.isContinuous()) rgb_out = rgb_out.clone();
    return true;
}



void Model::writeInput(const uint8_t* rgb_data) {
    uint8_t* dst = static_cast<uint8_t*>(input_mem->virt_addr);
 
    const size_t row_bytes = static_cast<size_t>(model_width) * model_channel;
    const size_t stride_px = input_attrs[0].w_stride;  // row length in pixels, may be padded
 
    if (stride_px == 0 || stride_px == static_cast<size_t>(model_width)) {
        std::memcpy(dst, rgb_data, row_bytes * model_height);
    } else {
        const size_t dst_row_bytes = stride_px * model_channel;
        for (int h = 0; h < model_height; ++h) {
            std::memcpy(dst + h * dst_row_bytes, rgb_data + h * row_bytes, row_bytes);
        }
    }
}
 
bool Model::readOutput(uint32_t index, OutputTensor& out) const {
    const rknn_tensor_attr& a = output_attrs[index];
    const rknn_tensor_mem*  m = output_mems[index];
    if (!m || !m->virt_addr || a.n_dims == 0) return false;

    // Obliczenia kroków i wymiarów dla pamięci sprzętowej NPU
    const size_t w      = a.dims[a.n_dims - 1];
    const size_t n      = a.n_elems;
    const size_t stride = (a.w_stride > w) ? a.w_stride : w;
    const size_t rows   = w ? n / w : 0;
    const uint8_t* base = static_cast<const uint8_t*>(m->virt_addr);

    // 1. Zachowujemy parametry skali na później bezpośrednio w strukturze
    const bool affine = (a.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    out.scale = affine ? a.scale : 1.0f;
    out.zp    = affine ? a.zp : 0;

    out.dims.assign(a.dims, a.dims + a.n_dims);
    
    // 2. Przygotowujemy wektor na natywne dane (INT8)
    out.data.resize(n);
    int8_t* dst = out.data.data();

    // 3. Kopiujemy dane pozbywając się wyrównania (w_stride), bez zmiany ich wartości
    for (size_t r = 0; r < rows; ++r) {
        if (a.type == RKNN_TENSOR_INT8 || a.type == RKNN_TENSOR_UINT8) {
            const int8_t* src = reinterpret_cast<const int8_t*>(base) + r * stride;
            for (size_t c = 0; c < w; ++c) {
                *dst++ = src[c];
            }
        } else if (a.type == RKNN_TENSOR_FLOAT32) {
            std::cerr << "Błąd: Oczekiwano wyjścia INT8, model zwrócił FLOAT32!\n";
            return false;
        } else {
            std::cerr << "Nieobsługiwany typ wyjścia: " << typeName(a.type) << "\n";
            return false;
        }
    }
    return true;
}
 
bool Model::run(const uint8_t* rgb_data, std::vector<OutputTensor>& results) {
    writeInput(rgb_data);
 
    const int ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_run failed! ret=" << ret << "\n";
        return false;
    }
 
    results.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        if (!readOutput(i, results[i])) return false;
    }
    return true;
}



bool Model::inference(const cv::Mat& bgr_frame,
                      std::vector<OutputTensor>& results,
                      LetterboxInfo* letterbox) {
    LetterboxInfo lb;
    cv::Mat rgb;
    if (!preprocess(bgr_frame, rgb, lb)) return false;
    if (letterbox) *letterbox = lb;
    return run(rgb.data, results);
}

Model::~Model() {
        if (ctx != 0) {
            rknn_destroy(ctx);
            ctx = 0;
        }
}
