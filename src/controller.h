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
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int16.hpp"
#include "geometry_msgs/msg/twist.h"
#include <geometry_msgs/msg/detail/twist__struct.hpp>

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
    inline static shared_timed_mutex mutexUpdateKDTree;
    inline static mutex mutexReplanCommitedPath;
    inline static mutex mutexUpdateGraph;
    inline static vector<DataForDebug> dataForDebug = {};
    inline static Parameters param;
    inline static vector<GraphEdge *> currentPath = {};
    inline static int currentIndexPath = -1;
    inline static VectorXd explorationPosition = VectorXd::Zero(3);
    inline static vector<RobotPose> commitedPath;
    inline static double measuredHeight;

    inline static std::thread lowLevelMovementThread;
    inline static std::thread replanOmegaThread;
    inline static std::thread updateGraphThread;
    inline static std::thread updateKDTreeThread;
    inline static std::thread transitionAlgThread;
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
    rclcpp::TimerBase::SharedPtr mainLoopTimer;

    CBFNavQuad();
    void endCallback(const std_msgs::msg::Int16::SharedPtr msg);
    void updatePose();
    void mainFunction();
    double getTime();
    RobotPose getRobotPose();
    void setTwist(VectorXd linearVelocity, double angularVelocity);
    void setLinearVelocity(VectorXd linearVelocity);
    static vector<vector<VectorXd>> getFrontierPoints();
    static vector<VectorXd> getLidarPointsSource(VectorXd position, double radius);
    static vector<VectorXd> getLidarPointsKDTree(VectorXd position, double radius);
    void lowLevelMovement();
    void replanCommitedPathCall();
    void replanCommitedPath();
    void updateGraphCall();
    void updateGraph();
    void updateKDTreeCall();
    void updateKDTree();
    void transitionAlg();


    void debug_addMessage(int counter, string msg);
    void debug_Store(int counter);
    void debug_generateManyPathsReport(int counter);
    
};


