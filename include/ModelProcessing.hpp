#pragma once

#include <cstdint>
#include <rknn_api.h>
#include <cstddef>
#include <vector>
#include <cstring>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp> 
#include <Types.hpp>


class Model {        
        private:
                rknn_context ctx = 0; //główny kontekst gdzie jest model
                rknn_input_output_num io_num{};
                
                std::vector<rknn_tensor_attr> input_attrs;
                std::vector<rknn_tensor_attr> output_attrs;
                

                int model_width = 0;
                int model_height = 0;
                int model_channel = 0;

                
                rknn_tensor_mem*              input_mem = nullptr;
                std::vector<rknn_tensor_mem*> output_mems;

                static std::vector<uint8_t> readFileData(const std::string& filename) {
                        //Odczyt plikó z wagami
                        std::ifstream file(filename, std::ios::binary | std::ios::ate);
                        if (!file.is_open()) return {};
        
                        std::streamsize size = file.tellg();
                        if (size <= 0) return {};
        
                        std::vector<uint8_t> buffer(static_cast<size_t>(size));
                        file.seekg(0, std::ios::beg);
                        file.read(reinterpret_cast<char*>(buffer.data()), size);
                        return buffer;
                }

                bool preprocess(const cv::Mat& bgr, cv::Mat& rgb_out, LetterboxInfo& lb) const;
                bool run(const uint8_t* rgb_data, std::vector<OutputTensor>& results);

                
                bool setupIoMemory();                                   
                // allocate + bind NPU buffers (once)
        
                void writeInput(const uint8_t* rgb_data);               
                // copy frame into the input buffer

                bool readOutput(uint32_t index, OutputTensor& out) const;  
                // dequantize output buffer -> float
                

        public:
                Model() = default;

                Model(const Model&) = delete;
                Model& operator=(const Model&) = delete;

                bool init(const std::string& model_path);

                bool inference(const cv::Mat& bgr_frame,
                   std::vector<OutputTensor>& results,
                   LetterboxInfo* letterbox = nullptr);

                int getModelWidth() const { return model_width; }
                int getModelHeight() const { return model_height; }
                int getModelChannel() const { return model_channel; }

                const std::vector<rknn_tensor_attr>& getOutputAttrs() const { return output_attrs; }

                ~Model();
};
