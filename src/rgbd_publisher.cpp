// Std
#include <chrono>
#include <memory>
#include <stdio.h>

// Ros
// #include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
// #include <image_transport/image_transport.hpp>
// Zed
#include <sl/Camera.hpp>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace sl;

class ZedPublisher : public rclcpp::Node{

    private:
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _rgb_pub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _depth_pub;
        // Create a ZED camera object
        Camera _zed;

        // Zed Camera Images
        sl::Mat _image, _depth; 
        // OpenCV Camera Images
        cv::Mat _cv_image, _cv_depth;

        // Camera Settings
        InitParameters _init_parameters;
        RuntimeParameters _runtime_parameters;

    public:
        
        ZedPublisher() : Node("zed_publisher"){
            _rgb_pub = this->create_publisher<sensor_msgs::msg::Image>("image", 10);
            _depth_pub = this->create_publisher<sensor_msgs::msg::Image>("depth/image", 10);
        }

        int init_camera(){
            // Set configuration parameters
            _init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE; // Use PERFORMANCE depth mode
            _init_parameters.coordinate_units = UNIT::MILLIMETER; // Use millimeter units (for depth measurements)

            // Set runtime parameters after opening the camera
            _runtime_parameters.sensing_mode = SENSING_MODE::STANDARD; // Use STANDARD sensing mode

            // Open the camera
            auto returned_state = _zed.open(_init_parameters);
            if (returned_state != ERROR_CODE::SUCCESS) {
                // cout << "Error " << returned_state << ", exit program." << endl;
                return EXIT_FAILURE;
            }

            _image = sl::Mat(zed.getCameraInformation().camera_configuration.resolution, MAT_TYPE::U8_C4);
            _depth = sl::Mat(zed.getCameraInformation().camera_configuration.resolution, MAT_TYPE::U8_C4);

            _cv_image = slMat2cvMat(_image);
            _cv_depth = slMat2cvMat(_depth);
            return EXIT_SUCCESS;
        }

        void publish(){
            while (_zed.isOpened()) {
                // A new image is available if grab() returns ERROR_CODE::SUCCESS
                if (_zed.grab(_runtime_parameters) == ERROR_CODE::SUCCESS) {
                    // Retrieve left image
                    _zed.retrieveImage(_image, VIEW::LEFT);
                    // Retrieve depth map. Depth is aligned on the left image
                    _zed.retrieveMeasure(_depth, MEASURE::DEPTH);

                    // cv::imshow("Image", cv_image);
                    cv::imshow("Depth", _cv_depth);
                    cv::waitKey(1);
                }
            }
            // Close the camera
            _zed.close();
            return EXIT_SUCCESS;
        }

        // Mapping between MAT_TYPE and CV_TYPE
        int getOCVtype(sl::MAT_TYPE type) {
            int cv_type = -1;
            switch (type) {
                case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
                case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
                case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
                case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
                case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
                case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
                case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
                case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
                default: break;
            }
            return cv_type;
        }

        /**
        * Conversion function between sl::Mat and cv::Mat
        **/
        cv::Mat slMat2cvMat(Mat& input) {
            // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
            // cv::Mat and sl::Mat will share a single memory structure
            return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
        }

}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    std::shared_ptr<ZedPublisher> zed_publisher = std::make_shared<ZedPublisher>();
    zed_publisher->init_camera();
    zed_publisher->publish();

    rclcpp::spin(zed_publisher);
    rclcpp::shutdown();
}