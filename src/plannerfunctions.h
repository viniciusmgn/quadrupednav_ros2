#pragma once

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <list>
#include <math.h>
#include <vector>
#include <random>
#include <memory>

using namespace std;
using namespace Eigen;

namespace CBFCirc
{

    struct Parameters
    {
        double boundingRadius = 0.25; // 0.3 0.35 0.25
        double boundingHeight = 1.25;
        double smoothingParam = 0.15; // 0.1 0.3 0.5 0.3

        double constantHeight = -0.03; // 0.8 -0.1725 -0.08
        double marginSafety = 0.4;       // 0.8
        double updateKDTreepRadius = 10.0;      // 3.0 5.0
        double sensingRadius = 3.0;      // 3.0 5.0

        double gainRobotYaw = 4.0;         // 2.0 4.0
        double gainTargetController = 0.4; // 0.2
        double alphaCBFPositive = 1.0;
        double alphaCBFNegative = 6.0;   // 7.5 //6
        double distanceMinBeta = 0.10; // 0.5 0.3 0.4 0.50 0.30 0.15
        double maxVelCircBeta = 1.25;  // 0.5 0.5
        double maxTotalVel = 0.3;
        double distanceMarginPlan = 0.05; // 0.20

        double deltaTimePlanner = 0.4;   // 0.1 0.2
        double maxTimePlanner = 300;     // 50 100 150
        double plannerReachError = 0.50; // 0.25
        double plannerOmegaPlanReachError = 0.30; // 0.25
        double acceptableRationPlanning = 2.0;
        double acceptableRatioChangeCirc = 0.7;

        int freqStoreDebug = 15;
        int freqReplanPath = 1; // 500
        int freqUpdateGraph = 1;
        int freqUpdateKDTree = 1; // 100
        int freqDisplayMessage = 10;

        double noMaxIterationsCorrectPoint = 40;
        double stepCorrectPoint = 0.1;
        double radiusCreateNode = 2.0; // 0.8 1.5
        double maxTimePlanConnectNode = 100; //50
        double acceptableMinDist=2.5;

        double minDistFilterKDTree = 0.15; // 0.3

        int sampleFactorStorePath = 15;
        int sampleFactorLidarSource = 15; //5

        double minAcceptableNegativeDist = -0.35; //-0.35


        int noMaxOptimizePath = 10;
        double upsampleMinPos = 0.01; //0.01
        double upsampleMinOri = 0.01; //0.01
        double vectorFieldAlpha = 2.0;
        double distCutoffCorrect = 1.0;
        int generateSimplePathDiv = 100;
        
        double distPathFreePlan = 0.30; //0.15
        int noIterationsCorrectPath = 8; //7
        double correctPathStep = 0.3; //0.15 //0.3
        int filterWindow = 10;

        double distGroupFrontierPoints = 1.0;

        double maxTimeSampleExploration = 80;
        double deltaTimeSampleExploration = 1.0; //0.5
        double minDistExploration = 0.8; //0.5
        int noTriesClosestPoint = 5;
        double distPathFreeGraph = 0.05;
        //VectorXd globalTargetPosition = vec3d(7, 0, -0.1725); // vec3d(7, 0, -0.1725)
        //VectorXd globalTargetPosition = vec3d(-7, 1, -0.1725);
        //VectorXd globalTargetPosition = vec3d(13.0, -32.0, 0.0);
        //VectorXd globalTargetPosition = vec3d(20.0, 32.0, 0.0);
        VectorXd globalTargetPosition = vec3d(20.0, 0.0, 0.0);
        //VectorXd globalTargetPosition = vec3d(6.0, 0.0, 0.0);

        double distanceMarginLowLevel = 0.10; // 0.20 0.15 0.25
    };

    struct DistanceResult
    {
        double safety;
        double distance;
        VectorXd gradSafetyPosition;
        double gradSafetyOrientation;
        VectorXd witnessDistance;
    };
    struct RadialDistanceResult
    {
        double halfSqDistance;
        VectorXd gradDistance;
    };

    struct RobotPose
    {
        VectorXd position;
        double orientation;
    };

    struct CBFCircControllerResult
    {
        VectorXd linearVelocity;
        double angularVelocity;
        DistanceResult distanceResult;
        bool feasible;
    };

    struct CBFControllerResult
    {
        VectorXd linearVelocity;
        double angularVelocity;
        DistanceResult distanceResult;
        bool feasible;
    };

    enum class PathState
    {
        sucess,
        unfeasible,
        timeout,
        empty
    };

    struct GeneratePathResult
    {
        vector<RobotPose> path;
        vector<VectorXd> pathGradSafetyPosition;
        vector<double> pathGradSafetyOrientation;
        vector<double> pathDistance;
        PathState pathState;
        double finalError;
    };

    struct GenerateManyPathsResult
    {
        vector<GeneratePathResult> pathResults;
        vector<Matrix3d> pathOmega;
        vector<double> pathLenghts;
        bool atLeastOnePathReached;
        double bestPathSize;
        Matrix3d bestOmega;
        GeneratePathResult bestPath;
    };

    struct VectorFieldResult
    {
        VectorXd linearVelocity;
        double angularVelocity;
        double distance;
        int index;
    };

    struct OptimizePathResult
    {
        vector<RobotPose> path;
        double minDist;
        double simplifyTime;
        double correctPathTime;
        double filterTime;
    };

    struct CorrectPathResult
    {
        vector<RobotPose> path;
        double minDist;
    };

    enum class MotionPlanningState
    {
        goingToGlobalGoal,
        pathToExploration,
        goingToExplore,
        planning,
        success,
        failure
    };

    typedef function<vector<VectorXd>(VectorXd, double)> MapQuerier;

    DistanceResult computeDist(vector<VectorXd> points, RobotPose pose, Parameters param);
    CBFCircControllerResult CBFCircController(RobotPose pose, VectorXd targetPosition, vector<VectorXd> neighborPoints, Matrix3d omega,
                                              Parameters param);
    GeneratePathResult CBFCircPlanOne(RobotPose startingPose, VectorXd targetPosition, MapQuerier querier, Matrix3d omega,
                                      double maxTime, double reachpointError, double deltaTime, Parameters param);
    double curveLength(vector<RobotPose> posePath);
    GenerateManyPathsResult CBFCircPlanMany(RobotPose startingPose, VectorXd targetPosition, MapQuerier querier,
                                            double maxTime, double reachpointError, double deltaTime, Parameters param);
    RadialDistanceResult computeDistRadial(vector<VectorXd> points, VectorXd position, double smoothingParam);
    VectorXd correctPoint(VectorXd point, vector<VectorXd> neighborPoints, Parameters param);
    bool pathFree(vector<RobotPose> path, MapQuerier querier, int initialIndex, int finalIndex, double distTol, Parameters param);
    vector<RobotPose> generateSimplePath(vector<RobotPose> originalPath, MapQuerier querier, int initialIndex, int finalIndex, Parameters param);
    CorrectPathResult correctPath(vector<RobotPose> originalPath, MapQuerier querier, Parameters param);
    OptimizePathResult optimizePath(vector<RobotPose> originalPath, MapQuerier querier, Parameters param);
    vector<RobotPose> upsample(vector<RobotPose> path, double minDistPos, double minDistOri);
    VectorFieldResult vectorField(RobotPose pose, vector<RobotPose> path, Parameters param);
    CBFControllerResult CBFController(RobotPose pose, VectorXd targetLinearVelocity, double targetAngularVelocity,
                                      vector<VectorXd> neighborPoints, Parameters param);

}
