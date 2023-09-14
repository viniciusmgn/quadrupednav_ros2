#pragma once

#include <chrono>
#include <fstream>
#include <sstream>

#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int16.hpp>
#include <geometry_msgs/msg/twist.hpp>

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

#include "./kdtree-cpp-master/kdtree.hpp"
#include "./kdtree-cpp-master/kdtree.cpp"
#include <thread>
#include <mutex>

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
#include "controller.h"

#include <boost/tokenizer.hpp>

using namespace std;
using namespace Eigen;
using namespace CBFCirc;
using std::placeholders::_1;
using namespace boost;

// std::string debugging_folder = "/home/vinicius/Desktop/matlab/unitree_planning";
std::string debugging_folder = "/ros_ws/cbf_debugging";

CBFNavQuad::CBFNavQuad()
    : Node("cbfnavquad"),
      node_handle_(std::shared_ptr<CBFNavQuad>(this, [](auto *) {})),
      image_transport_(node_handle_)
{
    // Initialize ROS variables
    pubBodyTwist = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
    pubEnd = this->create_publisher<std_msgs::msg::Int16>("endProgram", 10);
    subEnd = this->create_subscription<std_msgs::msg::Int16>(
        "endProgram", 10, std::bind(&CBFNavQuad::endCallback, this, _1));

    this->declare_parameter("frontier_height_range", 0.10); // 0.2
    this->declare_parameter("main_loop_interval_ms", 10);
    this->declare_parameter("low_level_movement_loop_sleep_ms", 10);
    this->declare_parameter("replanning_loop_sleep_ms", 1500);
    this->declare_parameter("update_graph_loop_sleep_ms", 5000);
    this->declare_parameter("update_kdtree_loop_sleep_ms", 50);
    this->declare_parameter("transition_loop_sleep_ms", 10);
    this->declare_parameter("target_x", 5.0);
    this->declare_parameter("target_y", 0.0);

    // Image transport
    debug_visualizer = image_transport_.advertise("frontier_debug", 1);

    // Octomap subscriber
    octomap_subscriber_ = this->create_subscription<octomap_msgs::msg::Octomap>(
        "octomap_binary", 1,
        std::bind(&CBFNavQuad::OctomapCallback, this, std::placeholders::_1));

    tfBuffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

    poseCallbackTimer = this->create_wall_timer(10ms, std::bind(&CBFNavQuad::updatePose, this));
    // mainLoopTimer = this->create_wall_timer(10ms, std::bind(&CBFNavQuad::mainFunction, this));

    // Initialize some global variables
    Global::startTime = now().seconds();
    Global::currentGoalPosition = Global::param.globalTargetPosition;
    Global::currentOmega = Matrix3d::Zero();
    VectorXd startingPosition = vec3d(0, 0, Global::param.constantHeight);
    Global::graph.addNode(startingPosition);

    // DEBUG TEST
    // debugTest();
    //

    // Initialize some threads
    Global::lowLevelMovementThread = thread(std::bind(&CBFNavQuad::lowLevelMovement, this));
    // Global::replanOmegaThread = thread(std::bind(&CBFNavQuad::replanCommitedPath, this));
    // Global::updateGraphThread = thread(std::bind(&CBFNavQuad::updateGraph, this));
    // Global::updateKDTreeThread = thread(std::bind(&CBFNavQuad::updateKDTree, this));
    Global::transitionAlgThread = thread(std::bind(&CBFNavQuad::transitionAlg, this));
    Global::wholeAlgorithmThread = thread(std::bind(&CBFNavQuad::wholeAlgorithm, this));
    Global::storeDataThread = thread(std::bind(&CBFNavQuad::storeData, this));
}

void CBFNavQuad::debugTest()
{

    Global::pointsKDTree = readCSV("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/dataEnvironments.csv", 3);

    // Build the KDTree
    Kdtree::KdNodeVector nodes;

    // Guarantee that it has at least one node
    vector<double> point(3);
    point[0] = VERYBIGNUMBER;
    point[1] = VERYBIGNUMBER;
    point[2] = VERYBIGNUMBER;
    nodes.push_back(Kdtree::KdNode(point));

    for (int i = 0; i < Global::pointsKDTree.size(); i++)
    {
        vector<double> point(3);
        point[0] = Global::pointsKDTree[i][0];
        point[1] = Global::pointsKDTree[i][1];
        point[2] = Global::pointsKDTree[i][2];
        nodes.push_back(Kdtree::KdNode(point));
    }

    Global::kdTree = new Kdtree::KdTree(&nodes, 2);

    //
    vector<vector<VectorXd>> frontierPoints = getFrontierPoints();
    //

    vector<VectorXd> graphNodes = readCSV("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/dataGraphNode.csv", 3);
    vector<VectorXd> graphEdges = readCSV("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/dataGraphEdge.csv", 3);

    for (int i = 1; i < graphNodes.size(); i++)
        Global::graph.addNode(graphNodes[i]);

    for (int i = 0; i < graphEdges.size(); i++)
    {
        int ind1 = (int)graphEdges[i][0];
        int ind2 = (int)graphEdges[i][1];
        double val = graphEdges[i][2];

        Global::graph.connect(Global::graph.nodes[ind1], Global::graph.nodes[ind2], val, Matrix3d::Zero(3, 3));
    }

    vector<VectorXd> explorationPoints = readCSV("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/dataExploration.csv", 3);

    vector<vector<VectorXd>> explorationPointsVec;

    for (int i = 0; i < explorationPoints.size(); i++)
        explorationPointsVec.push_back({explorationPoints[i]});

    //
    RobotPose pose;
    vector<VectorXd> poseData = readCSV("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/pose.csv", 4);

    pose.position = vec3d(poseData[0][0], poseData[0][1], poseData[0][2]);
    pose.orientation = poseData[0][3];

    // NewExplorationPointResult nepr = Global::graph.getNewExplorationPoint(
    //     pose,
    //     [this](VectorXd position, double radius)
    //     { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
    //     explorationPointsVec, Global::param, this->get_logger());

    GenerateManyPathsResult gmpr = CBFCircPlanMany(
        pose, Global::param.globalTargetPosition,
        [this](VectorXd position, double radius)
        { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
        Global::param.maxTimePlanner, Global::param.plannerOmegaPlanReachError,
        Global::param.deltaTimePlanner, Global::param);

    OptimizePathResult opr = optimizePath(
        gmpr.bestPath.path, [this](VectorXd position, double radius)
        { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
        Global::param);

    ofstream f;
    f.open("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/ppopt.csv", ofstream::trunc);

    for (int i = 0; i < opr.path.size(); i++)
        f << opr.path[i].position[0] << ";" << opr.path[i].position[1] << ";" << opr.path[i].position[2] << ";" << opr.path[i].orientation << std::endl;

    f.flush();

    ofstream g0, g1;
    // g.open("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/ppraw.csv", ofstream::trunc);
    // for (int i = 0; i < gmpr.bestPath.path.size(); i++)
    //     g << gmpr.bestPath.path[i].position[0] << ";" << gmpr.bestPath.path[i].position[1] << ";" << gmpr.bestPath.path[i].position[2] << ";" << gmpr.bestPath.path[i].orientation << std::endl;

    g0.open("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/ppraw0.csv", ofstream::trunc);
    for (int i = 0; i < gmpr.pathResults[0].path.size(); i++)
        g0 << gmpr.pathResults[0].path[i].position[0] << ";" << gmpr.pathResults[0].path[i].position[1] << ";" << gmpr.pathResults[0].path[i].position[2] << ";" << gmpr.pathResults[0].path[i].orientation << std::endl;

    g0.flush();

    g1.open("/home/vinicius/Desktop/matlab/unitree_planning/debugTests/ppraw1.csv", ofstream::trunc);
    for (int i = 0; i < gmpr.pathResults[1].path.size(); i++)
        g1 << gmpr.pathResults[1].path[i].position[0] << ";" << gmpr.pathResults[1].path[i].position[1] << ";" << gmpr.pathResults[1].path[i].position[2] << ";" << gmpr.pathResults[1].path[i].orientation << std::endl;

    g1.flush();
}

void CBFNavQuad::storeData()
{

    while (rclcpp::ok() && Global::continueAlgorithm)
    {

        if (Global::measured && Global::firstPlanCreated && Global::planningState != MotionPlanningState::planning && Global::generalCounter > 10)
        {
            if (Global::storeCounter % 4 == 0)
            {
                // Global::mutexFrontierPoints.lock();
                // Global::frontierPoints = getFrontierPoints();
                // Global::mutexFrontierPoints.unlock();
            }

            debug_Store(Global::generalCounter);

            Global::storeCounter++;
        }

        this_thread::sleep_for(std::chrono::milliseconds(1500));
    }
}

void CBFNavQuad::wholeAlgorithm()
{
    while (rclcpp::ok() && Global::continueAlgorithm)
    {

        double target_x = this->get_parameter("target_x").as_double();
        double target_y = this->get_parameter("target_y").as_double();
        // CBFCirc::Parameters::globalTargetPosition = vec3d(target_x, target_y, 0.0);

        if (Global::measured)
        {

            // CBFNavQuad::transitionAlgCall();

            if (Global::generalCounter % Global::param.freqDisplayMessage == 0 && (Global::planningState != MotionPlanningState::planning))
            {
                if (Global::planningState == MotionPlanningState::goingToGlobalGoal)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "-----GOING TO GLOBAL TARGET (ver85)------");
                }
                if (Global::planningState == MotionPlanningState::pathToExploration)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "-------PATH TO EXPLORATION-------");
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Point " << (Global::currentIndexPath + 1) << " of " << (Global::currentPath.size()));
                }
                if (Global::planningState == MotionPlanningState::goingToExplore)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "---------GOING TO EXPLORE--------");
                }
                RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "distobs = " << Global::distance);
                RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "distgoal = " << (getRobotPose().position - Global::currentGoalPosition).norm());
                RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "omega = " << getMatrixName(Global::currentOmega));
            }

            // if (Global::generalCounter % 1000 == 0)
            //     refreshWholeMapCall();

            // RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Update KD Tree");

            if (Global::generalCounter % 50 == 0)
                updateKDTreeCall();

            // RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Replan commited path");

            if (Global::generalCounter % 300 == 0 || Global::asynchronousPlan) // 150 500
            {
                Global::asynchronousPlan = false;
                replanCommitedPathCall();
            }

            // RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Update Graph Call");

            if (Global::generalCounter % 250 == 0)
                updateGraphCall();

            // DEBUG
            // if (Global::firstPlanCreated && (Global::generalCounter % Global::param.freqStoreDebug == 0) && (Global::planningState != MotionPlanningState::planning))
            //     debug_Store(Global::generalCounter);

            // DEBUG
            Global::generalCounter++;
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Step " << Global::generalCounter);
        }
        else
        {
            RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Not measured yet...");
        }

        this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void CBFNavQuad::endCallback(const std_msgs::msg::Int16::SharedPtr msg)
{

    int16_t ind = msg.get()->data;
    if (ind == 1)
    {
        Global::continueAlgorithm = false;
        if (!Global::dataPrinted)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "Request to quit program...");
            RCLCPP_INFO_STREAM(this->get_logger(), "Started printing data");

            ofstream file;
            debug_printAlgStateToMatlab(&file);
            RCLCPP_INFO_STREAM(this->get_logger(), "Debug data printed!");
            Global::dataPrinted = true;
        }
    }
}

void CBFNavQuad::updatePose()
{

    try
    {
        geometry_msgs::msg::TransformStamped t;
        // t = tfBuffer->lookupTransform("B1_154/odom_fast_lio", "B1_154/imu_link", tf2::TimePointZero);
        t = tfBuffer->lookupTransform("B1_154/odom_kiss_icp", "B1_154/base_link", tf2::TimePointZero);

        double px = t.transform.translation.x;
        double py = t.transform.translation.y;
        double pz = Global::param.constantHeight;
        double x = t.transform.rotation.x;
        double y = t.transform.rotation.y;
        double z = t.transform.rotation.z;
        double w = t.transform.rotation.w;

        double coshalfv = w;
        double sinhalfv = sqrt(x * x + y * y + z * z);

        if (z < 0)
            sinhalfv = -sinhalfv;

        Global::position << px, py, pz;
        Global::orientation = 2 * atan2(sinhalfv, coshalfv);
        Global::measuredHeight = t.transform.translation.z;

        if (!Global::measured)
            Global::param.constantHeight = Global::measuredHeight;

        Global::measured = true;
    }
    catch (const tf2::TransformException &ex)
    {
    }
}

void CBFNavQuad::mainFunction()
{

    if (rclcpp::ok() && Global::continueAlgorithm)
    {
        // Receive path
        double target_x = this->get_parameter("target_x").as_double();
        double target_y = this->get_parameter("target_y").as_double();
        // CBFCirc::Parameters::globalTargetPosition = vec3d(target_x, target_y, 0.0);

        if (Global::measured)
        {
            if (Global::generalCounter % Global::param.freqDisplayMessage == 0 && (Global::planningState != MotionPlanningState::planning))
            {
                if (Global::planningState == MotionPlanningState::goingToGlobalGoal)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "-----GOING TO GLOBAL TARGET (ver2)------");
                }
                if (Global::planningState == MotionPlanningState::pathToExploration)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "-------PATH TO EXPLORATION-------");
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Point " << (Global::currentIndexPath + 1) << " of " << (Global::currentPath.size()));
                }
                if (Global::planningState == MotionPlanningState::goingToExplore)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "---------GOING TO EXPLORE--------");
                }
                RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "distobs = " << Global::distance);
                RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "distgoal = " << (getRobotPose().position - Global::currentGoalPosition).norm());
                RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "omega = " << getMatrixName(Global::currentOmega));
            }

            // DEBUG
            if (Global::firstPlanCreated && (Global::generalCounter % Global::param.freqStoreDebug == 0) && (Global::planningState != MotionPlanningState::planning))
                debug_Store(Global::generalCounter);
            // DEBUG

            Global::generalCounter++;
        }
        else
        {
            RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Not measured yet...");
        }
    }
}

double CBFNavQuad::getTime()
{
    return now().seconds() - Global::startTime;
}

RobotPose CBFNavQuad::getRobotPose()
{
    RobotPose pose;
    pose.position = Global::position;
    pose.orientation = Global::orientation;
    return pose;
}

void CBFNavQuad::setTwist(VectorXd linearVelocity, double angularVelocity)
{

    double theta = getRobotPose().orientation;
    double ctheta = cos(theta);
    double stheta = sin(theta);

    Vector2d velTransformed;
    velTransformed << ctheta * linearVelocity[0] + stheta * linearVelocity[1], -stheta * linearVelocity[0] + ctheta * linearVelocity[1];

    geometry_msgs::msg::Twist twist;
    twist.linear.x = velTransformed[0];
    twist.linear.y = velTransformed[1];
    twist.linear.z = 0;
    twist.angular.x = 0;
    twist.angular.y = 0;
    twist.angular.z = angularVelocity;

    Global::desLinVelocity = linearVelocity;
    Global::desAngVelocity = angularVelocity;
    pubBodyTwist->publish(twist);
}

void CBFNavQuad::setLinearVelocity(VectorXd linearVelocity)
{

    double theta = getRobotPose().orientation;
    Vector3d normVelocity = linearVelocity.normalized();
    double angularVelocity = Global::param.gainRobotYaw * (cos(theta) * normVelocity[1] - sin(theta) * normVelocity[0]);

    setTwist(linearVelocity, angularVelocity);
}

vector<vector<VectorXd>> CBFNavQuad::getFrontierPoints()
{
    vector<vector<VectorXd>> frontierPoints = getFrontierPointsHeight(Global::measuredHeight);

    if (frontierPoints.size() == 0)
    {
        frontierPoints = getFrontierPointsHeight(Global::measuredHeight - 0.2);
        if (frontierPoints.size() == 0)
        {
            frontierPoints = getFrontierPointsHeight(Global::measuredHeight + 0.2);

            if (frontierPoints.size() == 0)
                RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Tried everything... was not able to get frontier points.");

            return frontierPoints;
        }
        else
            return frontierPoints;
    }
    else
        return frontierPoints;
}

vector<vector<VectorXd>> CBFNavQuad::getFrontierPointsHeight(double height)
{
    vector<vector<VectorXd>> frontierPoints;

    // Cannot proceed without a map
    if (octomap_ptr == nullptr)
    {
        RCLCPP_WARN(this->get_logger(), "Map is not initialized");
        return frontierPoints;
    }

    double rz_min = height - 0.2;
    double rz_max = height + 0.2;

    // Pixelize, with 2px buffer for boundary points
    double x_max, y_max, z_max, x_min, y_min, z_min;
    octomap_ptr->getMetricMax(x_max, y_max, z_max);
    octomap_ptr->getMetricMin(x_min, y_min, z_min);

    const double map_resolution = octomap_ptr->getResolution(), buffer_factor = 2.0;

    x_max += map_resolution * buffer_factor;
    y_max += map_resolution * buffer_factor;
    x_min -= map_resolution * buffer_factor;
    y_min -= map_resolution * buffer_factor;

    const size_t image_width =
                     static_cast<size_t>(std::ceil((x_max - x_min) / map_resolution)),
                 image_height =
                     static_cast<size_t>(std::ceil((y_max - y_min) / map_resolution));

    const size_t map_origin_x = static_cast<size_t>(std::ceil(-x_min / map_resolution)),
                 map_origin_y = static_cast<size_t>(std::ceil(-y_min / map_resolution));
    const cv::Point map_origin(map_origin_x, map_origin_y);

    octomap::point3d max_bounds(x_max, y_max, rz_max),
        min_bounds(x_min, y_min, rz_min);
    cv::Mat occupied_map(image_height, image_width, CV_8UC1, cv::Scalar(0)),
        free_map(image_height, image_width, CV_8UC1, cv::Scalar(0));

    // Mutex access to the tree
    {
        std::unique_lock<std::mutex> lock(octomap_lock);
        rclcpp::Time section = this->now();
        for (octomap::OcTree::leaf_bbx_iterator
                 it = octomap_ptr->begin_leafs_bbx(min_bounds, max_bounds),
                 end = octomap_ptr->end_leafs_bbx();
             it != end; ++it)
        {
            size_t x_coord =
                       static_cast<size_t>(std::ceil((it.getX() - x_min) / map_resolution)),
                   y_coord =
                       static_cast<size_t>(std::ceil((it.getY() - y_min) / map_resolution));

            // If logOdd > 0 -> Occupied. Otherwise free
            // Checks for overlapping free / occupied is not essential
            if (it->getLogOdds() > 0)
            {
                occupied_map.at<uint8_t>(y_coord, x_coord) = 255;
                free_map.at<uint8_t>(y_coord, x_coord) = 0;
            }
            else
            {
                if (occupied_map.at<uint8_t>(y_coord, x_coord) == 0)
                    free_map.at<uint8_t>(y_coord, x_coord) = 255;
            }
        }
    }

    FindFrontierPointResult ffpr;
    // Frontier point extraction
    {
        // rclcpp::Time section = this->now();
        ffpr = FindFrontierPoints(free_map, occupied_map, map_origin);
        // RCLCPP_DEBUG_STREAM(this->get_logger(), "Frontier idenfitication completed in "
        //                                             << MilliSecondsSinceTime(section)
        //                                             << " ms");
    }

    if (ffpr.frontiers.size() > 0)
    {
        // DEBUG Visualize before post-processing
        // VisualizeFrontierCall(free_map, occupied_map, ffpr);

        // Post-processing to convert points into real coordinates
        for (size_t i = 0; i < ffpr.frontiers.size(); ++i)
        {
            ffpr.frontiers[i].x = ffpr.frontiers[i].x * map_resolution + x_min;
            ffpr.frontiers[i].y = ffpr.frontiers[i].y * map_resolution + y_min;
        }

        // Process the information
        int idMax = 0;
        int idMin = 1000;
        for (int i = 0; i < ffpr.cluster_id.size(); i++)
        {
            idMax = ffpr.cluster_id[i] > idMax ? ffpr.cluster_id[i] : idMax;
            idMin = ffpr.cluster_id[i] < idMin ? ffpr.cluster_id[i] : idMin;
        }

        for (int i = 0; i <= idMax - idMin; i++)
        {
            vector<VectorXd> points = {};
            frontierPoints.push_back(points);
        }

        for (int i = 0; i < ffpr.frontiers.size(); i++)
        {
            VectorXd newPoint = VectorXd::Zero(3);
            newPoint << ffpr.frontiers[i].x, ffpr.frontiers[i].y, Global::param.constantHeight;
            frontierPoints[ffpr.cluster_id[i] - idMin].push_back(newPoint);
        }

        // Filter frontier points
        vector<vector<VectorXd>> frontierPointsFiltered = {};
        Global::mutexUpdateKDTree.lock_shared();
        for (int i = 0; i < frontierPoints.size(); i++)
        {
            double maxDist = 0;
            for (int j = 0; j < frontierPoints[i].size(); j++)
            {
                vector<VectorXd> points = getLidarPointsKDTree(frontierPoints[i][j], Global::param.sensingRadius);
                double dist = VERYBIGNUMBER;
                for (int k = 0; k < points.size(); k++)
                    dist = min(dist, (points[k] - frontierPoints[i][j]).norm());

                maxDist = max(maxDist, dist);
            }
            if (maxDist > max(Global::param.boundingRadius, Global::param.boundingHeight / 2))
                frontierPointsFiltered.push_back(frontierPoints[i]);
        }

        // Group points
        vector<VectorXd> centerPoint = {};
        for (int i = 0; i < frontierPointsFiltered.size(); i++)
        {
            VectorXd center = vec3d(0, 0, 0);
            for (int j = 0; j < frontierPointsFiltered[i].size(); j++)
                center += frontierPointsFiltered[i][j];

            center = center / ((double)frontierPointsFiltered[i].size());
            centerPoint.push_back(center);
        }

        vector<vector<int>> groups = {{0}};
        for (int i = 1; i < frontierPointsFiltered.size(); i++)
        {
            bool cont = true;
            int j = -1;
            double minDist;
            while (cont)
            {
                j++;
                minDist = VERYBIGNUMBER;
                for (int k = 0; k < groups[j].size(); k++)
                    minDist = min(minDist, (centerPoint[i] - centerPoint[groups[j][k]]).norm());

                cont = j + 1 < groups.size() && (minDist > Global::param.distGroupFrontierPoints);
            }

            if (minDist <= Global::param.distGroupFrontierPoints)
                groups[j].push_back(i);
            else
                groups.push_back({i});
        }

        vector<vector<VectorXd>> frontierPointsGrouped = {};

        for (int i = 0; i < groups.size(); i++)
        {
            frontierPointsGrouped.push_back({});
            for (int j = 0; j < groups[i].size(); j++)
                for (int k = 0; k < frontierPointsFiltered[groups[i][j]].size(); k++)
                    frontierPointsGrouped[i].push_back(frontierPointsFiltered[groups[i][j]][k]);
        }

        Global::mutexUpdateKDTree.unlock_shared();
        return frontierPointsGrouped;
        // return frontierPoints;
    }
    else
    {
        return {};
    }
}

vector<VectorXd> CBFNavQuad::getLidarPointsSource(VectorXd position, double radius)
{

    vector<VectorXd> points = {};
    double height_range = this->get_parameter("frontier_height_range").as_double();

    // Cannot proceed without a map
    if (octomap_ptr == nullptr)
    {
        RCLCPP_WARN(this->get_logger(), "Map is not initialized");
        return {};
    }

    int fact = Global::param.sampleFactorLidarSource;

    // Set bounds of the map to be extracted
    {
        std::unique_lock<std::mutex> lock(octomap_lock);
        octomap::point3d max_bounds(position[0] + radius, position[1] + radius, Global::measuredHeight + height_range),
            min_bounds(position[0] - radius, position[1] - radius, Global::measuredHeight - height_range);

        int k = 0;
        for (octomap::OcTree::leaf_bbx_iterator
                 it = octomap_ptr->begin_leafs_bbx(min_bounds, max_bounds),
                 end = octomap_ptr->end_leafs_bbx();
             it != end; ++it)
        {

            // If logOdd > 0 -> Occupied. Otherwise free
            if ((k % fact == 0) && (it->getLogOdds() > 0))
            {
                double x = it.getX(), y = it.getY(), z = it.getZ();
                // if (z >= Global::measuredHeight - height_range && z <= Global::measuredHeight + height_range)
                points.push_back(vec3d(x, y, Global::param.constantHeight));
            }

            if (it->getLogOdds() > 0)
                k++;
        }
    }

    // DONT FORGET TO REMOVE: add noise
    // if (abs(Global::generalCounter - 500) < 40)
    // {
    //     RobotPose pose = getRobotPose();

    //     for (int k = 0; k < 20; k++)
    //     {
    //         double X = pose.position[0] + 0.5 * cos(pose.orientation) + rand(-0.5, 0.5);
    //         double Y = pose.position[1] + 0.5 * sin(pose.orientation) + rand(-0.5, 0.5);
    //         double Z = pose.position[2];
    //         points.push_back(vec3d(X, Y, Z));
    //     }
    //     for (int k = 0; k < 20; k++)
    //     {
    //         double X = pose.position[0] - 0.5 * cos(pose.orientation) + rand(-0.5, 0.5);
    //         double Y = pose.position[1] - 0.5 * sin(pose.orientation) + rand(-0.5, 0.5);
    //         double Z = pose.position[2];
    //         points.push_back(vec3d(X, Y, Z));
    //     }
    // }

    return points;
}

vector<VectorXd> CBFNavQuad::getLidarPointsKDTree(VectorXd position, double radius)
{

    vector<VectorXd> points = {};

    if (Global::pointsKDTree.size() > 0)
    {

        vector<double> positionV(3);
        positionV[0] = position[0];
        positionV[1] = position[1];
        positionV[2] = position[2];

        Kdtree::KdNodeVector result;
        Global::kdTree->range_nearest_neighbors(positionV, radius, &result);

        for (int i = 0; i < result.size(); ++i)
        {
            VectorXd ptemp = vec3d(result[i].point[0], result[i].point[1], result[i].point[2]);
            points.push_back(ptemp);
        }
    }

    // vector<VectorXd> glps = getLidarPointsSource(position, radius);

    // for(int i=0; i < glps.size()/2; i++)
    //     points.push_back(glps[2*i]);

    return points;
}

void CBFNavQuad::lowLevelMovement()
{

    while (rclcpp::ok() && Global::continueAlgorithm)
    {
        if (Global::measured && Global::firstPlanCreated)
        {
            // RCLCPP_DEBUG(rclcpp::get_logger("low_level_logger"), "Low level call start");
            // RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Low level");

            if (Global::planningState != MotionPlanningState::planning && Global::commitedPath.size() > 1)
            {
                ////

                vector<VectorXd> pointsLidar = getLidarPointsSource(getRobotPose().position, Global::param.sensingRadius);

                VectorFieldResult vfr = vectorField(getRobotPose(), Global::commitedPath, Global::param);
                CBFControllerResult cccr = CBFController(getRobotPose(), vfr.linearVelocity, vfr.angularVelocity,
                                                         pointsLidar, Global::param);

                // Send the twist
                double multiplicative_factor = 1.5;

                // multiplicative_factor = min(max(1.4*(cccr.distanceResult.distance-0.05)/(1.0-0.05),0.0),1.4);

                setTwist(multiplicative_factor * cccr.linearVelocity, multiplicative_factor * cccr.angularVelocity);

                // setTwist(1.2 * cccr.linearVelocity, 1.2 * cccr.angularVelocity);

                // Refresh some variables
                Global::distance = cccr.distanceResult.distance;
                Global::safety = cccr.distanceResult.safety;
                Global::gradSafetyPosition = cccr.distanceResult.gradSafetyPosition;
                Global::gradSafetyOrientation = cccr.distanceResult.gradSafetyOrientation;
                Global::witnessDistance = cccr.distanceResult.witnessDistance;

                //
                // Global::distance = 0;
                // Global::safety = 0;
                // Global::gradSafetyPosition = vec3d(0,0,0);
                // Global::gradSafetyOrientation = 0;
                // Global::witnessDistance = vec3d(0,0,0);
            }
            else
            {
                setTwist(VectorXd::Zero(3), 0);
            }
            RCLCPP_DEBUG(rclcpp::get_logger("low_level_logger"), "Low level call end");
        }
        int sleep_interval = this->get_parameter("low_level_movement_loop_sleep_ms").as_int();
        this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
    }
}

void CBFNavQuad::replanCommitedPathCall()
{

    // Global::mutexReplanCommitedPath.lock();
    updateKDTreeCall();
    // Global::mutexUpdateKDTree.lock_shared();

    // DEBUG
    int counter = Global::generalCounter;
    debug_addMessage(counter, "Store event: start replanning commited path");
    // if (Global::firstPlanCreated)
    //     debug_Store(counter);
    // DEBUG

    auto start = high_resolution_clock::now();

    // std::bind(&CBFNavQuad::getLidarPointsKDTree, this)

    GenerateManyPathsResult gmpr = CBFCircPlanMany(
        getRobotPose(), Global::currentGoalPosition,
        [this](VectorXd position, double radius)
        { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
        Global::param.maxTimePlanner, Global::param.plannerOmegaPlanReachError,
        Global::param.deltaTimePlanner, Global::param);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    debug_addMessage(counter, "CBFCircPlanMany took " + std::to_string((double)duration.count() / E106) + " s");

    OptimizePathResult opr;
    if (gmpr.atLeastOnePathReached)
    {
        start = high_resolution_clock::now();

        opr = optimizePath(
            gmpr.bestPath.path, [this](VectorXd position, double radius)
            { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
            Global::param);
        Global::commitedPath = opr.path;
        Global::currentOmega = gmpr.bestOmega;

        debug_addMessage(counter, "CorrectPathTime: " + std::to_string(opr.correctPathTime) + " s");
        debug_addMessage(counter, "SimplifyTime: " + std::to_string(opr.simplifyTime) + " s");
        debug_addMessage(counter, "FilterTime: " + std::to_string(opr.filterTime) + " s");
        debug_addMessage(counter, "PathSize: " + std::to_string(opr.path.size()) + " points");
        debug_addMessage(counter, "minDist " + std::to_string(opr.minDist) + "");
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        debug_addMessage(counter, "optimizePath took " + std::to_string((double)duration.count() / E106) + " s");
    }

    Global::generateManyPathResult = gmpr;

    // DEBUG
    counter = Global::generalCounter;
    debug_addMessage(counter, "Store event: finished replanning commited path");
    debug_generateManyPathsReport(counter);
    debug_Store(counter);
    // DEBUG

    if (Global::generateManyPathResult.atLeastOnePathReached && opr.minDist > -0.20)
    {
        double lengthCurrentOmega;
        for (int j = 0; j < Global::generateManyPathResult.pathResults.size(); j++)
        {
            if (getMatrixNumber(Global::generateManyPathResult.pathOmega[j]) == getMatrixNumber(Global::currentOmega))
            {
                if (Global::generateManyPathResult.pathResults[j].pathState == PathState::sucess)
                    lengthCurrentOmega = Global::generateManyPathResult.pathLenghts[j];
                else
                    lengthCurrentOmega = VERYBIGNUMBER;
            }
        }
        if (Global::generateManyPathResult.bestPathSize < Global::param.acceptableRatioChangeCirc * lengthCurrentOmega)
        {
            Global::currentOmega = Global::generateManyPathResult.bestOmega;
            debug_addMessage(counter, "Circulation changed!");
        }
        else
            debug_addMessage(counter, "Circulation kept because of the ratio!");

        Global::firstPlanCreated = true;
    }
    else
    {
        // Transition condition
        RCLCPP_INFO_STREAM(this->get_logger(), "Failed to find path... plan to explore frontier!");
        debug_addMessage(counter, "Failed to find path... plan to explore frontier!");

        Global::planningState = MotionPlanningState::planning;
        vector<vector<VectorXd>> frontierPoints;

        // for (int i = 0; i < 3; i++)
        // {
        //     Global::mutexFrontierPoints.lock();
        //     frontierPoints = getFrontierPoints();
        //     Global::mutexFrontierPoints.unlock();
        //     std::this_thread::sleep_for(std::chrono::seconds(1));
        // }

        Global::mutexFrontierPoints.lock();
        frontierPoints = getFrontierPoints();
        Global::mutexFrontierPoints.unlock();

        while (frontierPoints.size() == 0)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "No frontier points found... trying again...");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            Global::mutexFrontierPoints.lock();
            frontierPoints = getFrontierPoints();
            Global::mutexFrontierPoints.unlock();
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "Start to update graph");
        updateGraphCall();
        RCLCPP_INFO_STREAM(this->get_logger(), "Graph updated");

        bool cont = true;

        while (cont)
        {
            Global::mutexUpdateGraph.lock();
            NewExplorationPointResult nepr = Global::graph.getNewExplorationPoint(
                getRobotPose(),
                [this](VectorXd position, double radius)
                { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
                frontierPoints, Global::param, this->get_logger());

            Global::explorationResult = nepr;
            Global::mutexUpdateGraph.unlock();

            if (nepr.success)
            {
                // Algorithm succesful
                RCLCPP_INFO_STREAM(this->get_logger(), "Frontier point selection successful!");
                debug_addMessage(counter, "Frontier point selection successful!");

                // Try to see if it can go to exploration point
                GenerateManyPathsResult gmpr = CBFCircPlanMany(
                    getRobotPose(), nepr.bestExplorationPosition,
                    [this](VectorXd position, double radius)
                    { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
                    Global::param.maxTimePlanner, Global::param.plannerOmegaPlanReachError,
                    Global::param.deltaTimePlanner, Global::param);

                if (gmpr.atLeastOnePathReached)
                {
                    RCLCPP_INFO_STREAM(this->get_logger(), "Skipped travelling to graph!");
                    debug_addMessage(counter, "Skipped travelling to graph");
                    Global::explorationPosition = nepr.bestExplorationPosition;
                    Global::currentGoalPosition = nepr.bestExplorationPosition;
                    Global::planningState = MotionPlanningState::goingToExplore;
                    OptimizePathResult opr = optimizePath(
                        gmpr.bestPath.path, [this](VectorXd position, double radius)
                        { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
                        Global::param);
                    Global::commitedPath = opr.path;
                    Global::currentOmega = gmpr.bestOmega;
                }
                else
                {
                    Global::currentPath = nepr.pathToExplorationPoint;
                    Global::currentIndexPath = 0;
                    Global::explorationPosition = nepr.bestExplorationPosition;
                    Global::currentGoalPosition = Global::currentPath[0]->nodeOut->position;

                    // DEBUG
                    debug_addMessage(counter, "Store event: beginning to travel path");
                    // debug_generateManyPathsReport(counter);
                    // debug_Store(counter);
                    // DEBUG

                    // Global::mutexReplanCommitedPath.unlock();
                    // Global::mutexUpdateKDTree.unlock_shared();
                    // CBFNavQuad::replanCommitedPathCall();
                    Global::asynchronousPlan = true;
                    // Global::mutexReplanCommitedPath.lock();
                    // Global::mutexUpdateKDTree.lock_shared();

                    Global::planningState = MotionPlanningState::pathToExploration;
                }
            }
            else
            {
                // Algorithm failed
                RCLCPP_INFO_STREAM(this->get_logger(), "Algorithm for finding new exploration points failed! Trying again");
                debug_Store(Global::generalCounter);
                // Global::planningState = MotionPlanningState::failure;
                // Global::continueAlgorithm = false;
                RCLCPP_INFO_STREAM(this->get_logger(), "Update KDTree");
                updateKDTreeCall(true);
                RCLCPP_INFO_STREAM(this->get_logger(), "Finished, trying again");
            }

            cont = !nepr.success;
        }
    }
    // Global::mutexUpdateKDTree.unlock_shared();
    // Global::mutexReplanCommitedPath.unlock();
}

void CBFNavQuad::replanCommitedPath()
{
    while (rclcpp::ok() && Global::continueAlgorithm)
    {

        if (Global::measured && (Global::generalCounter % Global::param.freqReplanPath == 0))
        {
            RCLCPP_DEBUG(rclcpp::get_logger("replanning_logger"), "Replanning call start");
            replanCommitedPathCall();
            RCLCPP_DEBUG(rclcpp::get_logger("replanning_logger"), "Replanning call end");
        }
        int sleep_interval = this->get_parameter("replanning_loop_sleep_ms").as_int();
        this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
    }
}

void CBFNavQuad::updateGraphCall()
{

    Global::mutexUpdateGraph.lock();
    Global::mutexUpdateKDTree.lock_shared();

    VectorXd currentPoint = getRobotPose().position;
    VectorXd correctedPoint = correctPoint(currentPoint, getLidarPointsKDTree(getRobotPose().position, Global::param.sensingRadius), Global::param);

    if ((Global::graph.getNeighborNodes(correctedPoint, Global::param.radiusCreateNode).size() == 0))
    {
        vector<double> distances;
        vector<int> indexes;
        vector<Matrix3d> omegas;

        vector<GraphNode *> nodeListSorted = Global::graph.getNearestNodeList(correctedPoint);

        int i = 0;
        bool cont = true;

        while (cont)
        {
            RobotPose pose;
            pose.position = nodeListSorted[i]->position;
            pose.orientation = 0;

            GenerateManyPathsResult gmpr = CBFCircPlanMany(
                pose, correctedPoint, [this](VectorXd position, double radius)
                { return CBFNavQuad::getLidarPointsKDTree(position, radius); },
                Global::param.maxTimePlanConnectNode, Global::param.plannerReachError,
                Global::param.deltaTimePlanner, Global::param);

            if (gmpr.atLeastOnePathReached)
            {
                indexes.push_back(i);
                omegas.push_back(gmpr.bestOmega);
                distances.push_back(gmpr.bestPathSize);
                cont = gmpr.bestPathSize > Global::param.acceptableMinDist;
            }

            i++;
            cont = cont && (i < nodeListSorted.size());
        }

        if (distances.size() > 0)
        {
            vector<int> ind = sortGiveIndex(distances);
            GraphNode *newNode = Global::graph.addNode(correctedPoint);
            Global::graph.connect(nodeListSorted[indexes[ind[0]]], newNode, distances[ind[0]], omegas[ind[0]]);

            // DEBUG
            double opfac = 100 * (1.0 - ((double)i) / ((double)nodeListSorted.size()));
            debug_addMessage(Global::generalCounter, "Graph updates with a new node! Optimization factor: " + std::to_string(opfac));
            //
        }
        else
        {
            // DEBUG
            debug_addMessage(Global::generalCounter, "Graph was not updated");
            //
        }
    }

    Global::mutexUpdateKDTree.unlock_shared();
    Global::mutexUpdateGraph.unlock();
}

void CBFNavQuad::updateGraph()
{
    while (rclcpp::ok() && Global::continueAlgorithm)
    {
        if (Global::measured && (Global::generalCounter % Global::param.freqUpdateGraph == 0))
        {
            RCLCPP_DEBUG(rclcpp::get_logger("graph_logger"), "Graph call start");
            updateGraphCall();
            RCLCPP_DEBUG(rclcpp::get_logger("graph_logger"), "Graph call end");
        }
        int sleep_interval = this->get_parameter("update_graph_loop_sleep_ms").as_int();
        this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
    }
}

void CBFNavQuad::refreshWholeMapCall()
{
    auto start = high_resolution_clock::now();

    Global::pointsKDTree = {};
    for (int k = 0; k < Global::graph.nodes.size(); k++)
    {
        vector<VectorXd> pointsFromLidar = getLidarPointsSource(Global::graph.nodes[k]->position, Global::param.updateKDTreepRadius);

        for (int i = 0; i < pointsFromLidar.size(); i++)
        {
            double minDist = VERYBIGNUMBER;
            int j = 0;
            bool cont = Global::pointsKDTree.size() > 0;
            while (cont)
            {
                minDist = min(minDist, (Global::pointsKDTree[j] - pointsFromLidar[i]).squaredNorm());
                j++;
                cont = (j < Global::pointsKDTree.size()) && (minDist > pow(Global::param.minDistFilterKDTree, 2));
            }

            if (minDist > pow(Global::param.minDistFilterKDTree, 2))
                Global::pointsKDTree.push_back(pointsFromLidar[i]);
        }
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    debug_addMessage(Global::generalCounter, "Updated the entire map. Took " + std::to_string(duration.count() / E106) + " s");
}

void CBFNavQuad::updateKDTreeCall(bool forceRemove)
{

    Global::mutexUpdateKDTree.lock();
    auto start = high_resolution_clock::now();

    vector<VectorXd> pointsFromLidar = getLidarPointsSource(getRobotPose().position, Global::param.updateKDTreepRadius);

    int debug_pointsAdded = 0;
    int debug_pointsRemoved = 0;

    for (int i = 0; i < pointsFromLidar.size(); i++)
    {
        double minDist = VERYBIGNUMBER;
        int j = 0;
        bool cont = Global::pointsKDTree.size() > 0;
        while (cont)
        {
            minDist = min(minDist, (Global::pointsKDTree[j] - pointsFromLidar[i]).squaredNorm());
            j++;
            cont = (j < Global::pointsKDTree.size()) && (minDist > pow(Global::param.minDistFilterKDTree, 2));
        }

        if (minDist > pow(Global::param.minDistFilterKDTree, 2))
        {
            Global::pointsKDTree.push_back(pointsFromLidar[i]);
            debug_pointsAdded++;
        }
    }

    // Check if some points should be removed
    if (forceRemove || (Global::generalCounter % 50 == 0 && Global::generalCounter > 0)) // 150
    {
        vector<VectorXd> pointsFromKDaround = getLidarPointsKDTree(getRobotPose().position, Global::param.updateKDTreepRadius);
        vector<VectorXd> pointsToRemove;

        for (int i = 0; i < pointsFromKDaround.size(); i++)
        {
            double mindDist = VERYBIGNUMBER;
            double minDist = VERYBIGNUMBER;
            int j = 0;
            bool cont = pointsFromLidar.size() > 0;
            while (cont)
            {
                minDist = min(minDist, (pointsFromLidar[j] - pointsFromKDaround[i]).squaredNorm());
                j++;
                cont = (j < pointsFromLidar.size()) && (minDist > pow(0.1, 2));
            }

            if (minDist > pow(0.2, 2)) // 0.1
            {
                pointsToRemove.push_back(pointsFromKDaround[i]);
                debug_pointsRemoved++;
            }
        }

        vector<int> indToRemove;

        for (int i = 0; i < pointsToRemove.size(); i++)
        {
            bool cont = Global::pointsKDTree.size() > 0;
            int j = 0;
            while (cont)
            {
                cont = (pointsToRemove[i] - Global::pointsKDTree[j]).norm() > 0.01 && j + 1 < Global::pointsKDTree.size();
                j++;
            }
            j--;
            if (j + 1 < Global::pointsKDTree.size())
                indToRemove.push_back(j);
        }

        sort(indToRemove.begin(), indToRemove.end());

        for (int i = indToRemove.size() - 1; i >= 0; i--)
            Global::pointsKDTree.erase(Global::pointsKDTree.begin() + indToRemove[i]);
    }

    // Build the KDTree
    Kdtree::KdNodeVector nodes;

    // Guarantee that it has at least one node
    vector<double> point(3);
    point[0] = VERYBIGNUMBER;
    point[1] = VERYBIGNUMBER;
    point[2] = VERYBIGNUMBER;
    nodes.push_back(Kdtree::KdNode(point));

    for (int i = 0; i < Global::pointsKDTree.size(); i++)
    {
        vector<double> point(3);
        point[0] = Global::pointsKDTree[i][0];
        point[1] = Global::pointsKDTree[i][1];
        point[2] = Global::pointsKDTree[i][2];
        nodes.push_back(Kdtree::KdNode(point));
    }

    Global::kdTree = new Kdtree::KdTree(&nodes, 2);

    Global::mutexUpdateKDTree.unlock();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    debug_addMessage(Global::generalCounter, "Updated KD Tree adding " + std::to_string(debug_pointsAdded) +
                                                 " points an removing " + std::to_string(debug_pointsRemoved) + ". Took " +
                                                 std::to_string(duration.count() / E106) + " s");
}

void CBFNavQuad::updateKDTree()
{
    while (rclcpp::ok() && Global::continueAlgorithm)
    {

        if (Global::measured && (Global::generalCounter % Global::param.freqUpdateKDTree == 0))
        {
            auto start = high_resolution_clock::now();
            RCLCPP_DEBUG(rclcpp::get_logger("kdtree_logger"), "kdtree call start");
            updateKDTreeCall();
            RCLCPP_DEBUG(rclcpp::get_logger("kdtree_logger"), "kdtree call end");
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            debug_addMessage(Global::generalCounter, "Took " + std::to_string((double)duration.count() / E106) + " s to update KDTree.");
        }
        int sleep_interval = this->get_parameter("update_kdtree_loop_sleep_ms").as_int();
        this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
    }
}

void CBFNavQuad::transitionAlgCall()
{
    bool pointReached = (getRobotPose().position - Global::currentGoalPosition).norm() <= Global::param.plannerReachError;

    if ((Global::planningState == MotionPlanningState::goingToGlobalGoal) && pointReached)
    {
        Global::planningState = MotionPlanningState::success;
        Global::continueAlgorithm = false;

        // DEBUG
        debug_addMessage(Global::generalCounter, "Success!");
        // DEBUG
    }

    if ((Global::planningState == MotionPlanningState::goingToExplore) && pointReached)
    {
        Global::planningState = MotionPlanningState::goingToGlobalGoal;
        Global::currentGoalPosition = Global::param.globalTargetPosition;

        Global::currentPath = {};
        Global::currentIndexPath = -1;
        Global::explorationPosition = VectorXd::Zero(3);

        replanCommitedPathCall();

        // DEBUG
        debug_addMessage(Global::generalCounter, "Reached exploration point. Going to global target!");
        // DEBUG
    }

    if ((Global::planningState == MotionPlanningState::pathToExploration) && pointReached)
    {
        if (Global::currentIndexPath == Global::currentPath.size() - 1)
        {
            Global::planningState = MotionPlanningState::goingToExplore;
            Global::currentGoalPosition = Global::explorationPosition;

            replanCommitedPathCall();

            // DEBUG
            debug_addMessage(Global::generalCounter, "Reached last point on the path. Going to explore a frontier...");
            // DEBUG
        }
        else
        {
            // DEBUG
            debug_addMessage(Global::generalCounter, "Reached point " + std::to_string(Global::currentIndexPath));
            // DEBUG

            Global::currentIndexPath++;
            Global::currentGoalPosition = Global::currentPath[Global::currentIndexPath]->nodeIn->position;
            Global::currentOmega = Global::currentPath[Global::currentIndexPath]->omega;
        }
    }
}

void CBFNavQuad::transitionAlg()
{
    while (rclcpp::ok() && Global::continueAlgorithm)
    {

        if (Global::measured)
        {
            // RCLCPP_DEBUG(rclcpp::get_logger("transition_logger"), "Transition call start");
            bool pointReached = (getRobotPose().position - Global::currentGoalPosition).norm() <= Global::param.plannerReachError;

            if ((Global::planningState == MotionPlanningState::goingToGlobalGoal) && pointReached)
            {
                Global::planningState = MotionPlanningState::success;
                Global::continueAlgorithm = false;

                RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Good!");

                cout << "Started printing data" << std::endl;
                ofstream file;
                CBFNavQuad::debug_printAlgStateToMatlab(&file);
                cout << "Debug data printed!" << std::endl;

                // DEBUG
                debug_addMessage(Global::generalCounter, "Success!");
                // DEBUG
            }

            if ((Global::planningState == MotionPlanningState::goingToExplore) && pointReached)
            {
                Global::planningState = MotionPlanningState::goingToGlobalGoal;
                Global::currentGoalPosition = Global::param.globalTargetPosition;

                Global::currentPath = {};
                Global::currentIndexPath = -1;
                Global::explorationPosition = VectorXd::Zero(3);

                Global::asynchronousPlan = true;

                // DEBUG
                debug_addMessage(Global::generalCounter, "Reached exploration point. Going to global target!");
                // DEBUG
            }

            if ((Global::planningState == MotionPlanningState::pathToExploration) && pointReached)
            {
                if (Global::currentIndexPath == Global::currentPath.size() - 1)
                {
                    Global::planningState = MotionPlanningState::goingToExplore;
                    Global::currentGoalPosition = Global::explorationPosition;

                    Global::asynchronousPlan = true;

                    // DEBUG
                    debug_addMessage(Global::generalCounter, "Reached last point on the path. Going to explore a frontier...");
                    // DEBUG
                }
                else
                {

                    Global::currentIndexPath++;
                    Global::currentGoalPosition = Global::currentPath[Global::currentIndexPath]->nodeIn->position;
                    Global::currentOmega = Global::currentPath[Global::currentIndexPath]->omega;

                    Global::asynchronousPlan = true;

                    // DEBUG
                    debug_addMessage(Global::generalCounter, "Reached point " + std::to_string(Global::currentIndexPath));
                    // DEBUG
                }
            }
            // RCLCPP_DEBUG(rclcpp::get_logger("transition_logger"), "Transition call end");
        }
        int sleep_interval = this->get_parameter("transition_loop_sleep_ms").as_int();
        this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
    }
}

// OCTOMAP

// Algorithm to extract the contours (inner & outer) of a given region
Contour CBFNavQuad::ExtractContour(const cv::Mat &free, const cv::Point &origin)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(free, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Select the contour containing the origin
    Contour relevant_contour;
    relevant_contour.valid = false;

    for (size_t i = 0; i < contours.size(); ++i)
    {
        bool origin_in_hole = false;

        const std::vector<cv::Point> &contour = contours[i];

        // Check if the origin is inside the outermost boundaries
        if (cv::pointPolygonTest(contour, origin, false) > 0)
        {
            // Check if the origin is not inside the holes (children) of the contour
            std::vector<std::vector<cv::Point>> children_contours;
            for (size_t j = 0; j < contours.size(); ++j)
                if (hierarchy[j][3] == static_cast<int>(i)) // Parent is the current contour
                    children_contours.push_back(contours[j]);

            for (const std::vector<cv::Point> &child_contour : children_contours)
            {
                // If the origin is inside a hole, then this is the incorrect contour
                if (cv::pointPolygonTest(child_contour, origin, false) > 0)
                    origin_in_hole = true;
                break;
            }

            // If the origin is not in any of the holes, then the current contour is
            // accurate
            if (!origin_in_hole)
            {
                relevant_contour.external = contour;
                relevant_contour.internal = children_contours;
                relevant_contour.valid = true;
            }
        }
    }

    if (!relevant_contour.valid)
    {
        RCLCPP_ERROR(this->get_logger(), "Contour cannot be set properly. Investigate");
    }

    return relevant_contour;
}

double CBFNavQuad::MilliSecondsSinceTime(const rclcpp::Time &start)
{
    return (this->now() - start).nanoseconds() * 1e-6;
}

void CBFNavQuad::OctomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg)
{
    // Reinitialize tree
    {
        std::unique_lock<std::mutex> lock(octomap_lock);
        // const double kResolution = msg.resolution;
        octomap_ptr.reset(
            dynamic_cast<octomap::OcTree *>(octomap_msgs::binaryMsgToMap(*msg)));
        octomap_ptr->expand();
    }
}

void CBFNavQuad::VisualizeFrontierCall(
    const cv::Mat &free_map,
    const cv::Mat &occupied_map,
    const FindFrontierPointResult &frontier)
{
    cv::Mat visual;
    cv::Mat zero_image(free_map.rows, free_map.cols, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Mat> channels{zero_image, free_map, occupied_map};
    cv::merge(channels, visual);

    // Add different color for each cluster
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> sample_255(0, 255);
    cv::Scalar cluster_color(255, 0, 255);
    size_t previous_cluster_id = 0;

    for (size_t i = 0; i < frontier.frontiers.size(); ++i)
    {
        // Resample color for a new cluster
        if (previous_cluster_id != frontier.cluster_id[i])
        {
            cluster_color = cv::Scalar(sample_255(rng), sample_255(rng), sample_255(rng));
        }

        cv::Point pt(frontier.frontiers[i].x, frontier.frontiers[i].y);
        cv::circle(visual, pt, 0, cluster_color, 1);

        previous_cluster_id = frontier.cluster_id[i];
    }

    // Correct the direction for better visualization
    cv::flip(visual, visual, 0);
    sensor_msgs::msg::Image::SharedPtr visMsg =
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", visual).toImageMsg();
    debug_visualizer.publish(visMsg);
}

FindFrontierPointResult CBFNavQuad::FindFrontierPoints(const cv::Mat &free_map, const cv::Mat &occupied_map, const cv::Point &map_origin)
{
    FindFrontierPointResult ffpr;
    Contour contour = ExtractContour(free_map, map_origin);

    // All external and internal points are at the boundary & are therefore valid
    // frontiers. Flatten the found contour
    std::vector<cv::Point> boundary_points;
    boundary_points.insert(boundary_points.end(), contour.external.begin(),
                           contour.external.end());
    for (const std::vector<cv::Point> &internal_contour : contour.internal)
        boundary_points.insert(boundary_points.end(), internal_contour.begin(),
                               internal_contour.end());

    if (boundary_points.size() == 0)
    {
        RCLCPP_WARN(this->get_logger(), "No frontier points found");
        ffpr.frontiers = {};
        return ffpr;
    }

    // Frontier points are the contours of free  space that are not adjacent to an
    // occupied location. Dilate the occupied cells with 3x3 kernel to extend the region
    // by 1 pixel. Alternative to querying 8-neighborhood for every contour
    cv::Mat kernel =
        cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
    cv::Mat occupied_contour;
    cv::dilate(occupied_map, occupied_contour, kernel);

    // Discontinuity indicates a cluster change for frontiers. Identified by a max.
    // coordinate distance of greater than 1
    cv::Point previous_pt = boundary_points[0];
    size_t cluster_id = 0;
    for (const auto &pt : boundary_points)
    {
        if (occupied_contour.at<uchar>(pt) != 255)
        {
            geometry_msgs::msg::Point coordinate;
            coordinate.x = pt.x;
            coordinate.y = pt.y;
            ffpr.frontiers.push_back(coordinate);

            // Increment cluster id if the frontier point is discontinuous
            // FIXME: If the cluster begins in the middle of a frontier line, the cluster
            // will be treated as two different clusters. Can be post-processed
            size_t distance =
                std::max(std::abs(previous_pt.x - pt.x), std::abs(previous_pt.y - pt.y));
            if (distance > 1)
                ++cluster_id;

            ffpr.cluster_id.push_back(cluster_id);
            previous_pt = pt;
        }
    }
    return ffpr;
}

// DEBUG

void CBFNavQuad::debug_Store(int counter)
{

    DataForDebug dfd;

    dfd.generalCounter = counter;
    dfd.timeStamp = getTime();
    dfd.position = getRobotPose().position;
    dfd.orientation = getRobotPose().orientation;
    dfd.desLinVelocity = Global::desLinVelocity;
    dfd.desAngVelocity = Global::desAngVelocity;
    dfd.distance = Global::distance;
    dfd.safety = Global::safety;
    dfd.gradSafetyPosition = Global::gradSafetyPosition;
    dfd.gradSafetyOrientation = Global::gradSafetyOrientation;
    dfd.witnessDistance = Global::witnessDistance;
    dfd.currentLidarPoints = getLidarPointsSource(getRobotPose().position, Global::param.updateKDTreepRadius);
    // Global::mutexUpdateKDTree.lock_shared();
    // dfd.currentLidarPointsKDTree = getLidarPointsKDTree(getRobotPose().position, Global::param.updateKDTreepRadius);
    // Global::mutexUpdateKDTree.unlock_shared();
    dfd.currentLidarPointsKDTree = {};
    dfd.currentGoalPosition = Global::currentGoalPosition;
    dfd.generateManyPathResult = Global::generateManyPathResult;
    dfd.currentOmega = Global::currentOmega;
    dfd.planningState = Global::planningState;
    dfd.graph = Global::graph;
    dfd.pointsKDTree = Global::pointsKDTree;
    dfd.pointsFrontier = Global::frontierPoints;
    dfd.currentPath = Global::currentPath;
    dfd.currentIndexPath = Global::currentIndexPath;
    dfd.explorationPosition = Global::explorationPosition;
    dfd.commitedPath = Global::commitedPath;
    dfd.explorationResult = Global::explorationResult;
    dfd.height = Global::measuredHeight;

    Global::dataForDebug.push_back(dfd);
}

void CBFNavQuad::debug_addMessage(int counter, string msg)
{
    Global::messages.push_back(std::to_string(counter) + ";" + msg);
}

void CBFNavQuad::debug_generateManyPathsReport(int counter)
{
    debug_addMessage(counter, "Path replanned!");

    for (int k = 0; k < Global::generateManyPathResult.pathResults.size(); k++)
    {
        string pathName = getMatrixName(Global::generateManyPathResult.pathOmega[k]);
        string errorToGoal = std::to_string(Global::generateManyPathResult.pathResults[k].finalError);
        string minimumError = std::to_string(Global::param.plannerOmegaPlanReachError);

        if (Global::generateManyPathResult.pathResults[k].pathState == PathState::sucess)
            debug_addMessage(counter, "Path " + pathName + " suceeded! Error to path was " + errorToGoal + " and minimum is " + minimumError);

        if (Global::generateManyPathResult.pathResults[k].pathState == PathState::unfeasible)
            debug_addMessage(counter, "Path " + pathName + " unfeasible!  Error to path was " + errorToGoal + " and minimum is " + minimumError);

        if (Global::generateManyPathResult.pathResults[k].pathState == PathState::timeout)
            debug_addMessage(counter, "Path " + pathName + " timeout! Error to path was " + errorToGoal + " but minimum is " + minimumError);
    }
}

void CBFNavQuad::debug_printAlgStateToMatlab(ofstream *f)
{

    time_t t = time(NULL);
    tm *timePtr = localtime(&t);
    string fname = "sim_cbf_unitree_ros2_";
    fname += std::to_string(timePtr->tm_mday) + "_" + std::to_string(timePtr->tm_mon + 1) + "_ts_" +
             std::to_string(timePtr->tm_hour) + "_" + std::to_string(timePtr->tm_min);

    // Update the file loader

    f->open(debugging_folder + "/fileloader.m", ofstream::trunc);

    // Write to load the files
    *f << "clc;" << std::endl;
    *f << "clear all;" << std::endl;
    *f << "dirData = '" << fname << "';" << std::endl;
    *f << "timeStamp = load([dirData '/timeStamp.csv']);" << std::endl;
    *f << "generalCounter = load([dirData '/generalCounter.csv']);" << std::endl;
    *f << "position = load([dirData '/position.csv']);" << std::endl;
    *f << "orientation = load([dirData '/orientation.csv']);" << std::endl;
    *f << "height = load([dirData '/height.csv']);" << std::endl;
    *f << "desLinVelocity = load([dirData '/desLinVelocity.csv']);" << std::endl;
    *f << "desAngVelocity = load([dirData '/desAngVelocity.csv']);" << std::endl;
    *f << "distance = load([dirData '/distance.csv']);" << std::endl;
    *f << "safety = load([dirData '/safety.csv']);" << std::endl;
    *f << "gradSafetyPosition = load([dirData '/gradSafetyPosition.csv']);" << std::endl;
    *f << "gradSafetyOrientation = load([dirData '/gradSafetyOrientation.csv']);" << std::endl;
    *f << "witnessDistance = load([dirData '/witnessDistance.csv']);" << std::endl;
    *f << "currentGoalPosition = load([dirData '/currentGoalPosition.csv']);" << std::endl;
    *f << "currentLidarPoints = processCell(load([dirData '/currentLidarPoints.csv']));" << std::endl;
    *f << "currentLidarPointsKDTree = processCell(load([dirData '/currentLidarPointsKDTree.csv']));" << std::endl;
    *f << "currentOmega = load([dirData '/currentOmega.csv']);" << std::endl;
    *f << "planningState = load([dirData '/planningState.csv']);" << std::endl;
    *f << "graphNodes = processCell(load([dirData '/graphNodes.csv']));" << std::endl;
    *f << "graphEdges = processCell(load([dirData '/graphEdges.csv']));" << std::endl;
    *f << "pointsKDTree = processCell(load([dirData '/pointsKDTree.csv']));" << std::endl;
    *f << "pointsFrontier = processCell(load([dirData '/pointsFrontier.csv']));" << std::endl;
    *f << "currentPath = processCell(load([dirData '/currentPath.csv']));" << std::endl;
    *f << "currentIndexPath = load([dirData '/currentIndexPath.csv']);" << std::endl;
    *f << "explorationPosition = load([dirData '/explorationPosition.csv']);" << std::endl;
    *f << "[messages,eventReplanFlag] = processMessageTable(readtable([dirData '/messages.csv']),generalCounter);" << std::endl;
    *f << "commitedPos = processCell(load([dirData '/commitedPos.csv']));" << std::endl;
    *f << "commitedOri = processCell(load([dirData '/commitedOri.csv']));" << std::endl;
    *f << "explorationResultData = processCell(load([dirData '/explorationResultData.csv']));" << std::endl;

    // Write planned paths
    vector<string> names = {};
    for (int k = 0; k < Global::generateManyPathResult.pathOmega.size(); k++)
        names.push_back(getMatrixName(Global::generateManyPathResult.pathOmega[k]));

    for (int k = 0; k < names.size(); k++)
    {
        *f << "plannedPos" << names[k] << " = processCell(load([dirData '/plannedPos" << names[k] << ".csv']));" << std::endl;
        *f << "plannedOri" << names[k] << " = processCell(load([dirData '/plannedOri" << names[k] << ".csv']));" << std::endl;
        *f << "plannedGradSafPos" << names[k] << " = processCell(load([dirData '/plannedGradSafPos" << names[k] << ".csv']));" << std::endl;
        *f << "plannedGradSafOri" << names[k] << " = processCell(load([dirData '/plannedGradSafOri" << names[k] << ".csv']));" << std::endl;
        *f << "plannedDistance" << names[k] << " = processCell(load([dirData '/plannedDistance" << names[k] << ".csv']));" << std::endl;
    }

    *f << "run " << fname << "/parameters.m;" << std::endl;
    f->flush();
    f->close();

    // Write to the parameters to a separate file
    boost::filesystem::create_directories(debugging_folder + "/" + fname);

    f->open(debugging_folder + "/" + fname + "/parameters.m", ofstream::trunc);
    *f << "param_boundingRadius =" << Global::param.boundingRadius << ";" << std::endl;
    *f << "param_boundingHeight =" << Global::param.boundingHeight << ";" << std::endl;
    *f << "param_smoothingParam =" << Global::param.smoothingParam << ";" << std::endl;
    f->flush();
    f->close();

    // Write the data
    vector<double> tempDouble;
    vector<VectorXd> tempVector;
    vector<vector<VectorXd>> tempVectorVector;

    // WRITE: timeStamp
    f->open(debugging_folder + "/" + fname + "/timeStamp.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back(Global::dataForDebug[i].timeStamp);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: generalCounter
    f->open(debugging_folder + "/" + fname + "/generalCounter.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].generalCounter);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: position
    f->open(debugging_folder + "/" + fname + "/position.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].position);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: orientation
    f->open(debugging_folder + "/" + fname + "/orientation.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].orientation);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: height
    f->open(debugging_folder + "/" + fname + "/height.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].height);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: desired linear velocity
    f->open(debugging_folder + "/" + fname + "/desLinVelocity.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].desLinVelocity);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: desired angular velocity
    f->open(debugging_folder + "/" + fname + "/desAngVelocity.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].desAngVelocity);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: distance
    f->open(debugging_folder + "/" + fname + "/distance.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].distance);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: safety
    f->open(debugging_folder + "/" + fname + "/safety.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].safety);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: gradient of safety on position
    f->open(debugging_folder + "/" + fname + "/gradSafetyPosition.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].gradSafetyPosition);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: gradient of safety on orientation
    f->open(debugging_folder + "/" + fname + "/gradSafetyOrientation.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].gradSafetyOrientation);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: witness points
    f->open(debugging_folder + "/" + fname + "/witnessDistance.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].witnessDistance);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: current lidar points
    f->open(debugging_folder + "/" + fname + "/currentLidarPoints.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].currentLidarPoints.size(); j++)
            tempVector.push_back(Global::dataForDebug[i].currentLidarPoints[j]);

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 3);
    f->flush();
    f->close();

    // WRITE: current lidar points from KD Tree
    f->open(debugging_folder + "/" + fname + "/currentLidarPointsKDTree.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].currentLidarPointsKDTree.size(); j++)
            tempVector.push_back(Global::dataForDebug[i].currentLidarPointsKDTree[j]);

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 3);
    f->flush();
    f->close();

    // WRITE: current goal position
    f->open(debugging_folder + "/" + fname + "/currentGoalPosition.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].currentGoalPosition);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: current matrix
    f->open(debugging_folder + "/" + fname + "/currentOmega.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)getMatrixNumber(Global::dataForDebug[i].currentOmega));

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: current state of the motion planning
    f->open(debugging_folder + "/" + fname + "/planningState.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].planningState);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: graph nodes
    f->open(debugging_folder + "/" + fname + "/graphNodes.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].graph.nodes.size(); j++)
            tempVector.push_back(Global::dataForDebug[i].graph.nodes[j]->position);

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 3);
    f->flush();
    f->close();

    // WRITE: graph edges
    f->open(debugging_folder + "/" + fname + "/graphEdges.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].graph.edges.size(); j++)
        {
            VectorXd edge = VectorXd::Zero(3);
            int inNode = Global::dataForDebug[i].graph.edges[j]->nodeIn->id;
            int outNode = Global::dataForDebug[i].graph.edges[j]->nodeOut->id;
            edge << (double)inNode, (double)outNode, Global::dataForDebug[i].graph.edges[j]->weight;
            tempVector.push_back(edge);
        }

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 3);
    f->flush();
    f->close();

    // WRITE: points Kd tree
    f->open(debugging_folder + "/" + fname + "/pointsKDTree.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].pointsKDTree.size(); j++)
            tempVector.push_back(Global::dataForDebug[i].pointsKDTree[j]);

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 3);
    f->flush();
    f->close();

    // WRITE: points frontier
    f->open(debugging_folder + "/" + fname + "/pointsFrontier.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].pointsFrontier.size(); j++)
            for (int k = 0; k < Global::dataForDebug[i].pointsFrontier[j].size(); k++)
                tempVector.push_back(Global::dataForDebug[i].pointsFrontier[j][k]);

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 3);
    f->flush();
    f->close();

    // WRITE: path in the graph
    f->open(debugging_folder + "/" + fname + "/currentPath.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].currentPath.size(); j++)
        {
            VectorXd ind = VectorXd::Zero(1);
            ind << (double)Global::dataForDebug[i].currentPath[j]->nodeIn->id;
            tempVector.push_back(ind);
        }

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 1);
    f->flush();
    f->close();

    // WRITE: current index in the path
    f->open(debugging_folder + "/" + fname + "/currentIndexPath.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        VectorXd ind = VectorXd::Zero(1);
        ind << (double)Global::dataForDebug[i].currentIndexPath;
        tempVector.push_back(ind);
    }

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: current exploration position
    f->open(debugging_folder + "/" + fname + "/explorationPosition.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].explorationPosition);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: commited position and orientation
    double fat = 3 * Global::param.sampleFactorStorePath;
    // fat = 1;

    f->open(debugging_folder + "/" + fname + "/commitedPos.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].commitedPath.size() / fat; j++)
            tempVector.push_back(Global::dataForDebug[i].commitedPath[fat * j].position);

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 3);
    f->flush();
    f->close();

    f->open(debugging_folder + "/" + fname + "/commitedOri.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].commitedPath.size() / fat; j++)
        {
            VectorXd data = VectorXd::Ones(1);
            data << Global::dataForDebug[i].commitedPath[fat * j].orientation;
            tempVector.push_back(data);
        }

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 1);
    f->flush();
    f->close();

    // WRITE: planned paths
    fat = Global::param.sampleFactorStorePath;

    for (int k = 0; k < names.size(); k++)
    {
        f->open(debugging_folder + "/" + fname + "/plannedPos" + names[k] + ".csv", ofstream::trunc);
        tempVectorVector = {};
        for (int i = 0; i < Global::dataForDebug.size(); i++)
        {
            tempVector = {};
            for (int j = 0; j < Global::dataForDebug[i].generateManyPathResult.pathResults[k].path.size() / fat; j++)
                tempVector.push_back(Global::dataForDebug[i].generateManyPathResult.pathResults[k].path[fat * j].position);

            tempVectorVector.push_back(tempVector);
        }
        printVectorVectorsToCSV(f, tempVectorVector, 3);
        f->flush();
        f->close();

        f->open(debugging_folder + "/" + fname + "/plannedOri" + names[k] + ".csv", ofstream::trunc);
        tempVectorVector = {};
        for (int i = 0; i < Global::dataForDebug.size(); i++)
        {
            tempVector = {};
            for (int j = 0; j < Global::dataForDebug[i].generateManyPathResult.pathResults[k].path.size() / fat; j++)
            {
                VectorXd data = VectorXd::Ones(1);
                data << Global::dataForDebug[i].generateManyPathResult.pathResults[k].path[fat * j].orientation;
                tempVector.push_back(data);
            }

            tempVectorVector.push_back(tempVector);
        }
        printVectorVectorsToCSV(f, tempVectorVector, 1);
        f->flush();
        f->close();

        f->open(debugging_folder + "/" + fname + "/plannedGradSafPos" + names[k] + ".csv", ofstream::trunc);
        tempVectorVector = {};
        for (int i = 0; i < Global::dataForDebug.size(); i++)
        {
            tempVector = {};
            for (int j = 0; j < Global::dataForDebug[i].generateManyPathResult.pathResults[k].path.size() / fat; j++)
                tempVector.push_back(Global::dataForDebug[i].generateManyPathResult.pathResults[k].pathGradSafetyPosition[fat * j]);

            tempVectorVector.push_back(tempVector);
        }
        printVectorVectorsToCSV(f, tempVectorVector, 3);
        f->flush();
        f->close();

        f->open(debugging_folder + "/" + fname + "/plannedGradSafOri" + names[k] + ".csv", ofstream::trunc);
        tempVectorVector = {};
        for (int i = 0; i < Global::dataForDebug.size(); i++)
        {
            tempVector = {};
            for (int j = 0; j < Global::dataForDebug[i].generateManyPathResult.pathResults[k].path.size() / fat; j++)
            {
                VectorXd data = VectorXd::Ones(1);
                data << Global::dataForDebug[i].generateManyPathResult.pathResults[k].pathGradSafetyOrientation[fat * j];
                tempVector.push_back(data);
            }

            tempVectorVector.push_back(tempVector);
        }
        printVectorVectorsToCSV(f, tempVectorVector, 1);
        f->flush();
        f->close();

        f->open(debugging_folder + "/" + fname + "/plannedDistance" + names[k] + ".csv", ofstream::trunc);
        tempVectorVector = {};
        for (int i = 0; i < Global::dataForDebug.size(); i++)
        {
            tempVector = {};
            for (int j = 0; j < Global::dataForDebug[i].generateManyPathResult.pathResults[k].path.size() / fat; j++)
            {
                VectorXd data = VectorXd::Ones(1);
                data << Global::dataForDebug[i].generateManyPathResult.pathResults[k].pathDistance[fat * j];
                tempVector.push_back(data);
            }

            tempVectorVector.push_back(tempVector);
        }
        printVectorVectorsToCSV(f, tempVectorVector, 1);
        f->flush();
        f->close();
    }

    // VectorXd point;
    // VectorXd selectedPointGraph;
    // double distToObstacle = VERYBIGNUMBER;
    // double distPointToGraph = VERYBIGNUMBER;
    // double distAlongGraph = VERYBIGNUMBER;
    // double distGraphToExploration = VERYBIGNUMBER;
    // double distExplorationToTarget = VERYBIGNUMBER;

    // WRITE: exploration data
    f->open(debugging_folder + "/" + fname + "/explorationResultData.csv", ofstream::trunc);
    tempVectorVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
    {
        tempVector = {};
        for (int j = 0; j < Global::dataForDebug[i].explorationResult.explorationPointDebugResult.size(); j++)
        {
            VectorXd data = VectorXd::Zero(15);
            data[0] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].point[0];
            data[1] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].point[1];
            data[2] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].point[2];
            data[3] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].selectedPointGraph[0];
            data[4] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].selectedPointGraph[1];
            data[5] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].selectedPointGraph[2];
            data[6] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].bestNodeToExploration[0];
            data[7] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].bestNodeToExploration[1];
            data[8] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].bestNodeToExploration[2];
            data[9] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].distToObstacle;
            data[10] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].distPointToGraph;
            data[11] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].distAlongGraph;
            data[12] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].distGraphToExploration;
            data[13] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].distExplorationToTarget;
            data[14] = Global::dataForDebug[i].explorationResult.explorationPointDebugResult[j].grade;

            tempVector.push_back(data);
        }

        tempVectorVector.push_back(tempVector);
    }
    printVectorVectorsToCSV(f, tempVectorVector, 15);
    f->flush();
    f->close();

    // WRITE: messages
    f->open(debugging_folder + "/" + fname + "/messages.csv", ofstream::trunc);
    for (int j = 0; j < Global::messages.size(); j++)
    {
        *f << Global::messages[j] << std::endl;
    }

    f->flush();
    f->close();
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CBFNavQuad>());

    if (Global::planningState == MotionPlanningState::success)
    {
        cout << "Started printing data" << std::endl;
        ofstream file;
        CBFNavQuad::debug_printAlgStateToMatlab(&file);
        cout << "Debug data printed!" << std::endl;
    }

    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "Outside loop");

    sleep(20);

    Global::lowLevelMovementThread.join();
    cout << "lowLevelMovementThread joined" << std::endl;
    Global::replanOmegaThread.join();
    cout << "replanOmegaThread joined" << std::endl;
    Global::updateGraphThread.join();
    cout << "updateGraphThread joined" << std::endl;
    Global::updateKDTreeThread.join();
    cout << "updateKDTreeThread joined" << std::endl;
    Global::transitionAlgThread.join();
    cout << "transitionAlgThread joined" << std::endl;

    rclcpp::shutdown();

    return 0;
}
