#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>  // VideoCapture, imwrite (opencv-mobile)

#include "ModelProcessing.hpp"
#include "PostProcess.hpp"
#include "ByteTrack.hpp"

// ---------------------------------------------------------------------------
// LED on GPIO2_A2 -> sysfs pin 66 (2*32 + 0*8 + 2)
// ---------------------------------------------------------------------------
namespace {

const int         LED_PIN   = 66;
const std::string GPIO_ROOT = "/sys/class/gpio";
const std::string LED_DIR   = GPIO_ROOT + "/gpio" + std::to_string(LED_PIN);

bool sysfsWrite(const std::string& path, const std::string& value)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << " (run as root?)\n";
        return false;
    }
    f << value;
    f.close();
    if (f.fail()) {
        std::cerr << "Write failed: " << path << " <- " << value << "\n";
        return false;
    }
    return true;
}

bool ledInit()
{
    if (access(LED_DIR.c_str(), F_OK) != 0) {          // export if needed
        if (!sysfsWrite(GPIO_ROOT + "/export", std::to_string(LED_PIN))) return false;
        usleep(100000);                                 // wait for gpio66 folder
    }
    return sysfsWrite(LED_DIR + "/direction", "out") &&
           sysfsWrite(LED_DIR + "/value", "0");         // start with LED off
}

void ledSet(bool on)
{
    sysfsWrite(LED_DIR + "/value", on ? "1" : "0");
}

void ledRelease()
{
    ledSet(false);
    sysfsWrite(GPIO_ROOT + "/unexport", std::to_string(LED_PIN));
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string model_path = (argc > 1) ? argv[1] : "/root/model/best-rv1106.rknn";

    Model model;
    if (!model.init(model_path)) {
        std::cerr << "Model init failed\n";
        return -1;
    }

    cv::VideoCapture cap;
    if (!cap.open(0)) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    // LED setup: if it fails we still run the tracker, just without the LED
    const bool led_ok = ledInit();
    if (!led_ok) std::cerr << "LED init failed - continuing without LED\n";
    bool led_on = false;

    PostprocessConfig pp_cfg;
    pp_cfg.conf_threshold = 0.1f;
    pp_cfg.nms_threshold  = 0.7f;

    BYTETracker tracker(30, 30, 0.5f, 0.6f, 0.8f);

    cv::Mat frame;
    std::vector<OutputTensor> outputs;
    LetterboxInfo letterbox;

    std::cout << "Start grabbing" << std::endl;
    for (int i = 0; i < 100; ++i)
    {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        const auto t0 = std::chrono::steady_clock::now();
        if (!model.inference(frame, outputs, &letterbox)) {
            std::cerr << "Inference failed\n";
            break;
        }
        const auto t1 = std::chrono::steady_clock::now();

        if (outputs.empty()) {
            std::cerr << "ERROR! model returned no output tensors\n";
            break;
        }

        const int model_w = static_cast<int>(std::round(frame.cols * letterbox.scale)) + 2 * letterbox.pad_x;
        const int model_h = static_cast<int>(std::round(frame.rows * letterbox.scale)) + 2 * letterbox.pad_y;

        // 1. Dekodowanie tensora i NMS
        PostprocessStats stats;
        PostprocessStats* stats_ptr = (i == 0) ? &stats : nullptr;

        std::vector<Detection> detections = postprocess(
            outputs[0],
            letterbox,
            model_w, model_h,
            frame.cols, frame.rows,
            pp_cfg,
            stats_ptr
        );

        // 2. Aktualizacja ByteTrack
        std::vector<STrackPtr> tracks = tracker.update(detections);

        // 2b. LED: on while at least one object is being tracked.
        //     Only touch sysfs when the state actually changes.
        const bool tracking = !tracks.empty();
        if (led_ok && tracking != led_on) {
            ledSet(tracking);
            led_on = tracking;
            std::cout << "LED " << (tracking ? "ON" : "OFF") << " (frame " << i << ")\n";
        }

        // 3. Rysowanie aktywnych śladów na klatce
        drawTracks(frame, tracks);

        const auto t2 = std::chrono::steady_clock::now();
        const double inf_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double post_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        std::cout << "frame " << i
                  << ": inf=" << inf_ms << " ms"
                  << ", post+track=" << post_ms << " ms"
                  << " | dets=" << detections.size()
                  << ", tracks=" << tracks.size() << "\n";

        // Zapis klatki co 30 iteracji (0, 30, 60, 90)
        if (i % 30 == 0) {
            const std::string filename = "/tmp/frame_" + std::to_string(i) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Saved " << filename << "\n";
        }

        // Diagnostyka dla pierwszej klatki
        if (i == 0) {
            for (size_t k = 0; k < outputs.size(); ++k) {
                std::cout << "output " << k << " dims=[";
                for (size_t d = 0; d < outputs[k].dims.size(); ++d)
                    std::cout << (d ? ", " : "") << outputs[k].dims[d];
                std::cout << "] zp=" << outputs[k].zp << " scale=" << outputs[k].scale << " first values:";
                const size_t n = std::min<size_t>(5, outputs[k].data.size());
                for (size_t v = 0; v < n; ++v) std::cout << " " << static_cast<int>(outputs[k].data[v]);
                std::cout << "\n";
            }
            std::cout << "letterbox: scale=" << letterbox.scale
                      << " pad_x=" << letterbox.pad_x
                      << " pad_y=" << letterbox.pad_y
                      << " (model=" << model_w << "x" << model_h << ")\n";
            std::cout << "postprocess stats: classes=" << stats.num_classes
                      << " anchors=" << stats.num_anchors
                      << " normalized=" << (stats.coords_normalized ? "yes" : "no")
                      << " max_coord=" << stats.max_coord
                      << " max_score=" << stats.max_score
                      << " candidates=" << stats.candidates << "\n";
        }
    }

    if (!frame.empty()) {
        cv::imwrite("/tmp/frame_last.jpg", frame);
    }

    cap.release();

    // LED off + release the pin
    if (led_ok) ledRelease();

    return 0;
}
