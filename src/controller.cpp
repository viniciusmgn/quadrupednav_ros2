#pragma once

#include <fstream>
#include <sstream>

#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int16.hpp"
#include "geometry_msgs/msg/twist.h"
#include <geometry_msgs/msg/detail/twist__struct.hpp>

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
// #include <octomap_with_query/neighbor_points.h>
// #include <octomap_with_query/frontier_points.h>

#include "./kdtree-cpp-master/kdtree.hpp"
#include "./kdtree-cpp-master/kdtree.cpp"
#include <thread>
#include <mutex>

#include "utils.h"
#include "utils.cpp"
#include "plannerfunctions.h"
#include "plannerfunctions.cpp"
#include "controller.h"

using namespace std;
using namespace Eigen;
using namespace CBFCirc;
using std::placeholders::_1;

CBFNavQuad::CBFNavQuad()
    : Node("cbfnavquad")
{
    // Initialize ROS variables

    pubBodyTwist = this->create_publisher<geometry_msgs::msg::Twist>("b1_gazebo/cmd_vel", 10);
    pubEnd = this->create_publisher<std_msgs::msg::Int16>("endProgram", 10);
    subEnd = this->create_subscription<std_msgs::msg::Int16>(
        "endProgram", 10, std::bind(&CBFNavQuad::endCallback, this, _1));

    tfBuffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

    poseCallbackTimer = this->create_wall_timer(100ms, std::bind(&CBFNavQuad::updatePose, this));
    mainLoopTimer = this->create_wall_timer(100ms, std::bind(&CBFNavQuad::mainFunction, this));

    // Initialize some global variables
    Global::startTime = now().seconds();
    Global::currentGoalPosition = Global::param.globalTargetPosition;
    Global::currentOmega = Matrix3d::Zero();
    VectorXd startingPosition = vec3d(0, 0, Global::param.constantHeight);
    Global::graph.addNode(startingPosition);

    // Initialize some threads
    Global::lowLevelMovementThread = thread(std::bind(&CBFNavQuad::lowLevelMovement, this));
    Global::replanOmegaThread = thread(std::bind(&CBFNavQuad::replanCommitedPath, this));
    Global::updateGraphThread = thread(std::bind(&CBFNavQuad::updateGraph, this));
    Global::updateKDTreeThread = thread(std::bind(&CBFNavQuad::updateKDTree, this));
    Global::transitionAlgThread = thread(std::bind(&CBFNavQuad::transitionAlg, this));
}

void CBFNavQuad::endCallback(const std_msgs::msg::Int16::SharedPtr msg)
{
    int16_t ind = msg.get()->data;
    if (ind == 1)
        Global::continueAlgorithm = false;
}

void CBFNavQuad::updatePose()
{

    try
    {
        geometry_msgs::msg::TransformStamped t;
        t = tfBuffer->lookupTransform("b1_gazebo/fast_lio", "b1_gazebo/base", tf2::TimePointZero);

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
        if (Global::measured)
        {

            if (Global::generalCounter % Global::param.freqDisplayMessage == 0 && (Global::planningState != MotionPlanningState::planning))
            {
                if (Global::planningState == MotionPlanningState::goingToGlobalGoal)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "-----GOING TO GLOBAL TARGET------");
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
    vector<vector<VectorXd>> frontierPoints;
    // octomap_with_query::frontier_points srv;

    // srv.request.z_min = Global::param.constantHeight - 0.2;
    // srv.request.z_max = Global::param.constantHeight + 0.2;

    // if (Global::frontierClient->call(srv))
    // {
    //     int idMax = 0;
    //     int idMin = 1000;
    //     for (int i = 0; i < srv.response.cluster_id.size(); i++)
    //     {
    //         idMax = srv.response.cluster_id[i] > idMax ? srv.response.cluster_id[i] : idMax;
    //         idMin = srv.response.cluster_id[i] < idMin ? srv.response.cluster_id[i] : idMin;
    //     }

    //     for (int i = 0; i <= idMax - idMin; i++)
    //     {
    //         vector<VectorXd> points = {};
    //         frontierPoints.push_back(points);
    //     }

    //     for (int i = 0; i < srv.response.frontiers.size(); i++)
    //     {
    //         VectorXd newPoint = VectorXd::Zero(3);
    //         newPoint << srv.response.frontiers[i].x, srv.response.frontiers[i].y, Global::param.constantHeight;
    //         frontierPoints[srv.response.cluster_id[i] - idMin].push_back(newPoint);
    //     }
    // }

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
    Global::mutexUpdateKDTree.unlock_shared();

    return frontierPointsFiltered;
}

vector<VectorXd> CBFNavQuad::getLidarPointsSource(VectorXd position, double radius)
{
    vector<VectorXd> points;

    // octomap_with_query::neighbor_points srv;

    // srv.request.radius = radius;
    // srv.request.query.x = position[0];
    // srv.request.query.y = position[1];
    // srv.request.query.z = position[2];

    // int fact = Global::param.sampleFactorLidarSource;

    // if (Global::neighborhClient->call(srv))
    // {
    //     for (int i = 0; i < srv.response.neighbors.size() / fact; i++)
    //     {
    //         double z = srv.response.neighbors[fact * i].z;
    //         if (z >= Global::measuredHeight - 0.10 && z <= Global::measuredHeight + 0.10)
    //         {
    //             VectorXd newPoint = vec3d(srv.response.neighbors[fact * i].x, srv.response.neighbors[fact * i].y, Global::param.constantHeight);
    //             points.push_back(newPoint);
    //         }
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

    return points;
}

void CBFNavQuad::lowLevelMovement()
{

    while (rclcpp::ok() && Global::continueAlgorithm)
    {
        if (Global::measured && Global::firstPlanCreated)
        {
            if (Global::planningState != MotionPlanningState::planning && Global::commitedPath.size() > 1)
            {
                vector<VectorXd> obsPoints = getLidarPointsSource(getRobotPose().position, Global::param.sensingRadius);
                VectorFieldResult vfr = vectorField(getRobotPose(), Global::commitedPath, Global::param);
                CBFControllerResult cccr = CBFController(getRobotPose(), vfr.linearVelocity, vfr.angularVelocity,
                                                         obsPoints, Global::param);

                // Send the twist
                setTwist(1.2 * cccr.linearVelocity, 1.2 * cccr.angularVelocity);

                // Refresh some variables
                Global::distance = cccr.distanceResult.distance;
                Global::safety = cccr.distanceResult.safety;
                Global::gradSafetyPosition = cccr.distanceResult.gradSafetyPosition;
                Global::gradSafetyOrientation = cccr.distanceResult.gradSafetyOrientation;
                Global::witnessDistance = cccr.distanceResult.witnessDistance;
            }
            else
            {
                setTwist(VectorXd::Zero(3), 0);
            }
        }
    }
}

void CBFNavQuad::replanCommitedPathCall()
{
    Global::mutexReplanCommitedPath.lock();
    updateKDTreeCall();
    Global::mutexUpdateKDTree.lock_shared();

    Global::generateManyPathResult = CBFCircPlanMany(getRobotPose(), Global::currentGoalPosition, getLidarPointsKDTree,
                                                     Global::param.maxTimePlanner, Global::param.plannerOmegaPlanReachError,
                                                     Global::param.deltaTimePlanner, Global::param);

    if (Global::generateManyPathResult.atLeastOnePathReached)
        Global::commitedPath = optimizePath(Global::generateManyPathResult.bestPath.path, getLidarPointsKDTree, Global::param);

    // DEBUG
    int counter = Global::generalCounter;
    debug_addMessage(counter, "Store event: replanning commited path");
    debug_generateManyPathsReport(counter);
    debug_Store(counter);
    // DEBUG

    if (Global::generateManyPathResult.atLeastOnePathReached)
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
        cout << "Failed to find path... plan to explore frontier!";
        debug_addMessage(counter, "Failed to find path... plan to explore frontier!");

        Global::planningState = MotionPlanningState::planning;
        vector<vector<VectorXd>> frontierPoints = getFrontierPoints();
        while (frontierPoints.size() == 0)
        {
            cout << "No frontier points found... trying again...";
            std::this_thread::sleep_for(std::chrono::seconds(5));
            frontierPoints = getFrontierPoints();
        }

        updateGraphCall();
        Global::mutexUpdateGraph.lock();
        NewExplorationPointResult nepr = Global::graph.getNewExplorationPoint(getRobotPose(), getLidarPointsKDTree,
                                                                              frontierPoints, Global::param);
        Global::mutexUpdateGraph.unlock();

        if (nepr.success)
        {
            // Algorithm succesful
            debug_addMessage(counter, "Frontier point selection successful... replanning path");

            Global::currentPath = nepr.pathToExplorationPoint;
            Global::currentIndexPath = 0;
            Global::explorationPosition = nepr.bestExplorationPosition;
            Global::currentGoalPosition = Global::currentPath[0]->nodeOut->position;

            // DEBUG
            debug_addMessage(counter, "Store event: beginning to travel path");
            debug_generateManyPathsReport(counter);
            debug_Store(counter);
            // DEBUG

            Global::mutexReplanCommitedPath.unlock();
            Global::mutexUpdateKDTree.unlock_shared();
            replanCommitedPathCall();
            Global::mutexReplanCommitedPath.lock();
            Global::mutexUpdateKDTree.lock_shared();

            Global::planningState = MotionPlanningState::pathToExploration;
        }
        else
        {
            // Algorithm failed
            cout << "Algorithm for finding new exploration points failed! Algorithm halted!";
            Global::planningState = MotionPlanningState::failure;
            Global::continueAlgorithm = false;
        }
    }
    Global::mutexUpdateKDTree.unlock_shared();
    Global::mutexReplanCommitedPath.unlock();
}

void CBFNavQuad::replanCommitedPath()
{
    while (rclcpp::ok() && Global::continueAlgorithm)
        if (Global::measured && (Global::generalCounter % Global::param.freqReplanPath == 0))
            replanCommitedPathCall();
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

        int i = 0;
        bool cont = true;

        while (cont)
        {
            RobotPose pose;
            pose.position = Global::graph.nodes[i]->position;
            pose.orientation = 0;

            GenerateManyPathsResult gmpr = CBFCircPlanMany(pose, correctedPoint, getLidarPointsKDTree,
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
            cont = cont && (i < Global::graph.nodes.size());
        }

        if (distances.size() > 0)
        {
            vector<int> ind = sortGiveIndex(distances);
            GraphNode *newNode = Global::graph.addNode(correctedPoint);
            Global::graph.connect(Global::graph.nodes[indexes[ind[0]]], newNode, distances[ind[0]], omegas[ind[0]]);

            // DEBUG
            debug_addMessage(Global::generalCounter, "Graph updates with a new node!");
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
        if (Global::measured && (Global::generalCounter % Global::param.freqUpdateGraph == 0))
            updateGraphCall();
}

void CBFNavQuad::updateKDTreeCall()
{

    Global::mutexUpdateKDTree.lock();

    vector<VectorXd> pointsFromLidar = getLidarPointsSource(getRobotPose().position, Global::param.sensingRadius);

    int debug_pointsAdded = 0;

    for (int i = 0; i < pointsFromLidar.size(); i++)
    {
        double minDist = VERYBIGNUMBER;
        for (int j = 0; j < Global::pointsKDTree.size(); j++)
            minDist = min(minDist, (Global::pointsKDTree[j] - pointsFromLidar[i]).squaredNorm());

        if (minDist > pow(Global::param.minDistFilterKDTree, 2))
        {
            Global::pointsKDTree.push_back(pointsFromLidar[i]);
            debug_pointsAdded++;
        }
    }

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
    debug_addMessage(Global::generalCounter, "Updated KD Tree with " + std::to_string(debug_pointsAdded) + " points");
}

void CBFNavQuad::updateKDTree()
{
    while (rclcpp::ok() && Global::continueAlgorithm)
        if (Global::measured && (Global::generalCounter % Global::param.freqUpdateKDTree == 0))
            updateKDTreeCall();
}

void CBFNavQuad::transitionAlg()
{
    while (rclcpp::ok() && Global::continueAlgorithm)
    {
        if (Global::measured)
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
    }
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
    dfd.currentLidarPoints = getLidarPointsSource(getRobotPose().position, Global::param.sensingRadius);
    Global::mutexUpdateKDTree.lock_shared();
    dfd.currentLidarPointsKDTree = getLidarPointsKDTree(getRobotPose().position, Global::param.sensingRadius);
    Global::mutexUpdateKDTree.unlock_shared();
    dfd.currentGoalPosition = Global::currentGoalPosition;
    dfd.generateManyPathResult = Global::generateManyPathResult;
    dfd.currentOmega = Global::currentOmega;
    dfd.planningState = Global::planningState;
    dfd.graph = Global::graph;
    dfd.pointsKDTree = Global::pointsKDTree;
    dfd.pointsFrontier = getFrontierPoints();
    dfd.currentPath = Global::currentPath;
    dfd.currentIndexPath = Global::currentIndexPath;
    dfd.explorationPosition = Global::explorationPosition;
    dfd.commitedPath = Global::commitedPath;

    Global::dataForDebug.push_back(dfd);
}

void CBFNavQuad::debug_addMessage(int counter, string msg)
{
    Global::messages.push_back(std::to_string(counter) + ";" + msg);
}

void CBFNavQuad::debug_generateManyPathsReport(int counter)
{
    debug_addMessage(counter, "Omega replanned!");
    for (int k = 0; k < Global::generateManyPathResult.pathResults.size(); k++)
    {
        string pathName = getMatrixName(Global::generateManyPathResult.pathOmega[k]);
        if (Global::generateManyPathResult.pathResults[k].pathState == PathState::sucess)
            debug_addMessage(counter, "Path " + pathName + " suceeded!");

        if (Global::generateManyPathResult.pathResults[k].pathState == PathState::unfeasible)
            debug_addMessage(counter, "Path " + pathName + " unfeasible!");

        if (Global::generateManyPathResult.pathResults[k].pathState == PathState::timeout)
        {
            string errorToGoal = std::to_string(Global::generateManyPathResult.pathResults[k].finalError);
            string minimumError = std::to_string(Global::param.plannerOmegaPlanReachError);
            debug_addMessage(counter, "Path " + pathName + " timeout! Error to path was " + errorToGoal + " but minimum is " + minimumError);
        }
    }
}

void debug_printAlgStateToMatlab(ofstream *f)
{

    time_t t = time(NULL);
    tm *timePtr = localtime(&t);
    string fname = "sim_cbf_unitree_ros2_";
    fname += std::to_string(timePtr->tm_mday) + "_" + std::to_string(timePtr->tm_mon + 1) + "_ts_" +
             std::to_string(timePtr->tm_hour) + "_" + std::to_string(timePtr->tm_min);

    // Update the file loader

    f->open("/home/vinicius/Desktop/matlab/unitree_planning/fileloader.m", ofstream::trunc);

    // Write to load the files
    *f << "clc;" << std::endl;
    *f << "clear all;" << std::endl;
    *f << "dirData = '" << fname << "';" << std::endl;
    *f << "timeStamp = load([dirData '/timeStamp.csv']);" << std::endl;
    *f << "generalCounter = load([dirData '/generalCounter.csv']);" << std::endl;
    *f << "position = load([dirData '/position.csv']);" << std::endl;
    *f << "orientation = load([dirData '/orientation.csv']);" << std::endl;
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
    *f << "messages = processMessageTable(readtable([dirData '/messages.csv']),generalCounter);" << std::endl;
    *f << "commitedPos = processCell(load([dirData '/commitedPos.csv']));" << std::endl;
    *f << "commitedOri = processCell(load([dirData '/commitedOri.csv']));" << std::endl;

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
    boost::filesystem::create_directories("/home/vinicius/Desktop/matlab/unitree_planning/" + fname);

    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/parameters.m", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/timeStamp.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back(Global::dataForDebug[i].timeStamp);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: generalCounter
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/generalCounter.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].generalCounter);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: position
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/position.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].position);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: orientation
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/orientation.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].orientation);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: desired linear velocity
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/desLinVelocity.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].desLinVelocity);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: desired angular velocity
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/desAngVelocity.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].desAngVelocity);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: distance
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/distance.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].distance);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: safety
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/safety.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].safety);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: gradient of safety on position
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/gradSafetyPosition.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].gradSafetyPosition);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: gradient of safety on orientation
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/gradSafetyOrientation.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].gradSafetyOrientation);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: witness points
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/witnessDistance.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].witnessDistance);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: current lidar points
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/currentLidarPoints.csv", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/currentLidarPointsKDTree.csv", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/currentGoalPosition.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].currentGoalPosition);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: current matrix
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/currentOmega.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)getMatrixNumber(Global::dataForDebug[i].currentOmega));

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: current state of the motion planning
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/planningState.csv", ofstream::trunc);
    tempDouble = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempDouble.push_back((double)Global::dataForDebug[i].planningState);

    printVectorsToCSV(f, tempDouble);
    f->flush();
    f->close();

    // WRITE: graph nodes
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/graphNodes.csv", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/graphEdges.csv", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/pointsKDTree.csv", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/pointsFrontier.csv", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/currentPath.csv", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/currentIndexPath.csv", ofstream::trunc);
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
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/explorationPosition.csv", ofstream::trunc);
    tempVector = {};
    for (int i = 0; i < Global::dataForDebug.size(); i++)
        tempVector.push_back(Global::dataForDebug[i].explorationPosition);

    printVectorsToCSV(f, tempVector);
    f->flush();
    f->close();

    // WRITE: commited position and orientation
    double fat = 3 * Global::param.sampleFactorStorePath;
    // fat = 1;

    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/commitedPos.csv", ofstream::trunc);
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

    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/commitedOri.csv", ofstream::trunc);
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
        f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/plannedPos" + names[k] + ".csv", ofstream::trunc);
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

        f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/plannedOri" + names[k] + ".csv", ofstream::trunc);
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

        f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/plannedGradSafPos" + names[k] + ".csv", ofstream::trunc);
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

        f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/plannedGradSafOri" + names[k] + ".csv", ofstream::trunc);
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

        f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/plannedDistance" + names[k] + ".csv", ofstream::trunc);
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

    // WRITE: messages
    f->open("/home/vinicius/Desktop/matlab/unitree_planning/" + fname + "/messages.csv", ofstream::trunc);
    for (int j = 0; j < Global::messages.size(); j++)
        *f << Global::messages[j] << std::endl;
    f->flush();
    f->close();
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CBFNavQuad>());
    rclcpp::shutdown();

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

    ofstream file;
    debug_printAlgStateToMatlab(&file);
    return 0;
}