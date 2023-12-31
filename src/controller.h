#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <list>
#include <math.h>
#include <vector>
#include <random>
#include <memory>
#include <functional>
#include <boost/multi_array.hpp>
#include <typeinfo>
#include <boost/filesystem.hpp>
#include <mutex>
#include <shared_mutex>
#include "./kdtree-cpp-master/kdtree.hpp"
#include <thread>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int16.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <cv_bridge/cv_bridge.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/conversions.h>
#include <image_transport/image_transport.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

#include "utils.h"
#include "utils.cpp"
#include "plannerfunctions.h"
#include "plannerfunctions.cpp"
#include "graph.h"
#include "graph.cpp"

using namespace std;
using namespace Eigen;
using namespace CBFCirc;

// STRUCT AND CLASSES

class DataForDebug
{
public:
    double timeStamp;
    int generalCounter;
    Vector3d position;
    double orientation;
    VectorXd desLinVelocity;
    double desAngVelocity;
    double distance;
    double safety;
    VectorXd gradSafetyPosition;
    double gradSafetyOrientation;
    VectorXd witnessDistance;
    vector<VectorXd> currentLidarPoints;
    vector<VectorXd> currentLidarPointsKDTree;
    VectorXd currentGoalPosition;
    GenerateManyPathsResult generateManyPathResult;
    Matrix3d currentOmega;
    MotionPlanningState planningState;
    Graph graph;
    vector<VectorXd> pointsKDTree;
    vector<vector<VectorXd>> pointsFrontier;
    vector<GraphEdge *> currentPath;
    int currentIndexPath;
    VectorXd explorationPosition;
    vector<RobotPose> commitedPath;
    NewExplorationPointResult explorationResult;
    double height;
};

class Global
{
public:
    inline static double startTime = 0;
    inline static VectorXd position = VectorXd::Zero(3);
    inline static double orientation = 0;
    inline static VectorXd desLinVelocity = VectorXd::Zero(3);
    inline static double desAngVelocity = 0;
    inline static int generalCounter = 0;
    inline static int storeCounter = 0;
    inline static bool measured = false;
    inline static double distance = 0;
    inline static double safety = 0;
    inline static VectorXd gradSafetyPosition = VectorXd::Zero(3);
    inline static double gradSafetyOrientation = 0;
    inline static VectorXd witnessDistance = VectorXd::Zero(3);
    inline static bool continueAlgorithm = true;
    inline static VectorXd currentGoalPosition = VectorXd::Zero(3);
    inline static GenerateManyPathsResult generateManyPathResult;
    inline static Matrix3d currentOmega;
    inline static MotionPlanningState planningState = MotionPlanningState::goingToGlobalGoal;
    inline static bool firstPlanCreated = false;
    inline static vector<string> messages = {};
    inline static Graph graph;
    inline static Kdtree::KdTree *kdTree;
    inline static vector<VectorXd> pointsKDTree = {};
    inline static vector<vector<VectorXd>> frontierPoints = {};

    inline static shared_timed_mutex mutexUpdateKDTree;
    inline static mutex mutexUpdateGraph;
    inline static mutex mutexFrontierPoints;

    inline static vector<DataForDebug> dataForDebug = {};
    inline static Parameters param;
    inline static vector<GraphEdge *> currentPath = {};
    inline static int currentIndexPath = -1;
    inline static VectorXd explorationPosition = VectorXd::Zero(3);
    inline static vector<RobotPose> commitedPath;
    inline static double measuredHeight;
    inline static bool dataPrinted = false;
    inline static NewExplorationPointResult explorationResult;

    inline static std::thread lowLevelMovementThread;
    inline static std::thread transitionAlgThread;
    inline static std::thread wholeAlgorithmThread;
    inline static std::thread storeDataThread;

    inline static bool asynchronousPlan = false;

    inline static vector<double> updateKDTreeCallTime = {};
    inline static vector<double> replanCommitedPathCallTime = {};
    inline static vector<double> updateGraphCallTime = {};
};

struct ContourResult
{
    std::vector<cv::Point> external;              // Points at the outermost boundary
    std::vector<std::vector<cv::Point>> internal; // Points at the boundary of holes
    bool valid;
};

struct FindFrontierPointResult
{
    vector<int> cluster_id;
    vector<geometry_msgs::msg::Point> frontiers;
};

class CBFNavQuad : public rclcpp::Node
{
public:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pubBodyTwist;
    rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr pubEnd;
    rclcpp::Subscription<std_msgs::msg::Int16>::SharedPtr subEnd;
    std::shared_ptr<tf2_ros::TransformListener> tfListener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tfBuffer;
    rclcpp::TimerBase::SharedPtr poseCallbackTimer;

    CBFNavQuad();
    void storeData();
    void wholeAlgorithm();
    void endCallback(const std_msgs::msg::Int16::SharedPtr msg);
    void updatePose();
    double getTime();
    RobotPose getRobotPose();
    void setTwist(VectorXd linearVelocity, double angularVelocity);
    void setLinearVelocity(VectorXd linearVelocity);
    vector<vector<VectorXd>> getFrontierPoints();
    vector<vector<VectorXd>> getFrontierPointsHeight(double height);
    vector<VectorXd> getLidarPointsSource(VectorXd position, double radius);
    vector<VectorXd> getLidarPointsKDTree(VectorXd position, double radius);
    void lowLevelMovement();
    void replanCommitedPathCall();
    void updateGraphCall();
    void updateKDTreeCall(bool forceRemove = false);
    void transitionAlg();
    void refreshWholeMapCall();
    ContourResult extractContour(const cv::Mat &free, const cv::Point &origin);
    void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg);
    double milliSecondsSinceTime(const rclcpp::Time &start);
    void visualizeFrontierCall(const cv::Mat &free_map, const cv::Mat &occupied_map,
                               const FindFrontierPointResult &frontier);
    FindFrontierPointResult findFrontierPoints(const cv::Mat &free_map, const cv::Mat &occupied_map, const cv::Point &map_origin);

    // DEBUG
    void debug_addMessage(int counter, string msg);
    void debug_Store(int counter);
    void debug_generateManyPathsReport(int counter);
    static void debug_printAlgStateToMatlab(ofstream *f);
    void debugTest();

private:
    // Storage and mutex lock for the latest map received
    std::unique_ptr<octomap::OcTree> octomap_ptr;
    std::mutex octomap_lock;

    // Image transport for debug
    rclcpp::Node::SharedPtr node_handle_;
    image_transport::ImageTransport image_transport_;
    image_transport::Publisher debug_visualizer; // Debug visualization

    // Subscribers & services
    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_subscriber_;
};
