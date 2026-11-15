#include <csignal>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

// GPIO2_A2 -> 2*32 + (0*8 + 2) = 66
static const int GPIO_PIN = 66;
static const std::string GPIO_ROOT = "/sys/class/gpio";

static volatile sig_atomic_t running = 1;
static void onSignal(int) { running = 0; }

static bool writeFile(const std::string &path, const std::string &value) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << " (are you root?)" << std::endl;
        return false;
    }
    f << value;
    f.close();
    if (f.fail()) {
        std::cerr << "Write failed: " << path << " <- " << value << std::endl;
        return false;
    }
    return true;
}

int main() {
    signal(SIGINT, onSignal);   // Ctrl+C
    signal(SIGTERM, onSignal);

    const std::string pin = std::to_string(GPIO_PIN);
    const std::string gpioDir = GPIO_ROOT + "/gpio" + pin;

    // 1. Export the pin (skip if already exported)
    if (access(gpioDir.c_str(), F_OK) != 0) {
        if (!writeFile(GPIO_ROOT + "/export", pin)) return 1;
        usleep(100000); // give the kernel time to create the gpio66 folder
    }

    // 2. Set direction to output
    if (!writeFile(gpioDir + "/direction", "out")) return 1;

    // 3. Blink: 1 second ON, 1 second OFF
    bool on = false;
    while (running) {
        on = !on;
        writeFile(gpioDir + "/value", on ? "1" : "0");
        std::cout << "LED " << (on ? "ON" : "OFF") << std::endl;
        sleep(1);
    }

    // 4. Clean up: LED off and release the pin
    writeFile(gpioDir + "/value", "0");
    writeFile(GPIO_ROOT + "/unexport", pin);
    std::cout << "Done." << std::endl;
    return 0;
}
