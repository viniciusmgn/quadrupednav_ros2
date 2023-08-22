#pragma once

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <list>
#include <math.h>
#include <vector>
#include <random>
#include <memory>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace Eigen;

#include "plannerfunctions.h"
#include "utils.h"

namespace CBFCirc
{
    DistanceResult computeDist(vector<VectorXd> points, RobotPose pose, Parameters param)
    {
        DistanceResult dr;
        vector<double> safety;
        vector<VectorXd> gradientsPosition;
        vector<VectorXd> gradientsOrientation;

        double cosang = cos(pose.orientation);
        double sinang = sin(pose.orientation);
        double x, y, z;

        dr.distance = VERYBIGNUMBER;
        dr.witnessDistance = VectorXd::Zero(3);

        for (int i = 0; i < points.size(); i++)
        {
            // Transform point

            x = points[i][0] - pose.position[0];
            y = points[i][1] - pose.position[1];
            z = points[i][2] - pose.position[2];
            VectorXd pointTrans = vec3d(cosang * x + sinang * y, -sinang * x + cosang * y, z);
            SafetyResult sr = safetyCylinder(pointTrans, param.boundingRadius, param.boundingHeight);

            // Compute safety
            safety.push_back(sr.safety);

            // Compute distance and the witness points
            double tempDist = signedDistCylinder(pointTrans, param.boundingRadius, param.boundingHeight);
            if (tempDist < dr.distance)
            {
                dr.distance = tempDist;
                dr.witnessDistance = points[i];
            }

            // Compute gradient of the safety on position

            x = sr.gradSafety[0];
            y = sr.gradSafety[1];
            z = sr.gradSafety[2];
            VectorXd gradSafetyPosition = vec3d(-(cosang * x - sinang * y), -(sinang * x + cosang * y), -z);
            gradientsPosition.push_back(gradSafetyPosition);

            // Compute gradient of the safety on orientation
            VectorXd pointRotated = vec3d(pointTrans[1], -pointTrans[0], 0);
            gradientsOrientation.push_back(sr.gradSafety.transpose() * pointRotated);
        }

        SoftSelectMinResult ssmrPosition = softSelectMin(3, safety, gradientsPosition, param.smoothingParam);
        SoftSelectMinResult ssmrOrientation = softSelectMin(1, safety, gradientsOrientation, param.smoothingParam);

        dr.safety = ssmrPosition.softMin - param.marginSafety;
        dr.gradSafetyPosition = ssmrPosition.selected;
        dr.gradSafetyOrientation = ssmrOrientation.selected[0];

        return dr;
    }

    CBFCircControllerResult CBFCircController(RobotPose pose, VectorXd targetPosition, vector<VectorXd> neighborPoints, Matrix3d omega, Parameters param)
    {
        DistanceResult dr = computeDist(neighborPoints, pose, param);

        VectorXd vd3d = -param.gainTargetController * (pose.position - targetPosition);
        VectorXd vd = VectorXd::Zero(2);
        vd << vd3d[0], vd3d[1];

        MatrixXd restvw = MatrixXd::Zero(1, 3);
        restvw(0, 0) = -param.gainRobotYaw * sin(pose.orientation);
        restvw(0, 1) = param.gainRobotYaw * cos(pose.orientation);
        restvw(0, 2) = -1;
        MatrixXd H1 = MatrixXd::Zero(3, 3);
        H1(0, 0) = 1;
        H1(1, 1) = 1;
        H1(2, 2) = min(max(dr.distance, 0.0), param.sensingRadius);
        MatrixXd H2 = restvw.transpose() * restvw;

        VectorXd ud = vectorVertStack(vd, 0);

        MatrixXd H = 2 * (H1 + H2); // 2 * (H1 + 4 * H2)
        VectorXd f = -2 * ud;
        MatrixXd A = MatrixXd::Zero(1, 3);
        A << dr.gradSafetyPosition[0], dr.gradSafetyPosition[1], dr.gradSafetyOrientation;
        VectorXd b = VectorXd::Zero(1);

        double cbfConst;
        if (dr.distance - param.distanceMarginPlan > 0)
            cbfConst = -param.alphaCBFPositive * (dr.distance - param.distanceMarginPlan);
        else
            cbfConst = -param.alphaCBFNegative * (dr.distance - param.distanceMarginPlan);

        b << cbfConst;

        // Add circulation is omega != 0
        if (abs(omega(0, 1)) + abs(omega(1, 2)) + abs(omega(2, 0)) >= VERYSMALLNUMBER)
        {
            MatrixXd A2 = MatrixXd::Zero(1, 3);
            VectorXd rotVec = omega * dr.gradSafetyPosition;
            A2 << rotVec[0], rotVec[1], 0;
            A = matrixVertStack(A, A2);
            double circConst = param.maxVelCircBeta * (1 - dr.distance / param.distanceMinBeta);
            b = vectorVertStack(b, circConst);
        }

        VectorXd u = solveQP(H, f, A, b);

        CBFCircControllerResult cccr;
        cccr.distanceResult = dr;
        cccr.linearVelocity = VectorXd::Zero(3);

        if (u.rows() > 0)
        {
            if (u.norm() >= param.maxTotalVel)
                u = param.maxTotalVel * u / (u.norm());

            cccr.linearVelocity << u[0], u[1], 0;
            cccr.angularVelocity = u[2];
            cccr.feasible = true;
        }
        else
        {
            cccr.angularVelocity = 0;
            cccr.feasible = false;
        }

        return cccr;
    }

    GeneratePathResult CBFCircPlanOne(RobotPose startingPose, VectorXd targetPosition, MapQuerier querier, Matrix3d omega,
                                      double maxTime, double reachpointError, double deltaTime, Parameters param)
    {
        GeneratePathResult gpr;
        RobotPose pose = startingPose;
        double time = 0;
        bool cont = true;

        gpr.pathState = PathState::sucess;
        gpr.path = {};
        gpr.pathGradSafetyPosition = {};
        gpr.pathGradSafetyOrientation = {};
        gpr.pathDistance = {};

        double dt;

        while (cont)
        {
            CBFCircControllerResult cccr = CBFCircController(pose, targetPosition, querier(pose.position, param.sensingRadius), omega, param);
            if (cccr.feasible)
            {
                gpr.path.push_back(pose);
                gpr.pathGradSafetyPosition.push_back(cccr.distanceResult.gradSafetyPosition);
                gpr.pathGradSafetyOrientation.push_back(cccr.distanceResult.gradSafetyOrientation);
                gpr.pathDistance.push_back(cccr.distanceResult.distance);

                if(cccr.distanceResult.distance>0.05)
                    dt = deltaTime;
                else 
                {
                    if(cccr.distanceResult.distance>0)
                        dt = deltaTime/2;
                    else
                        dt = deltaTime/4; 
                }   


                pose.position += cccr.linearVelocity * dt;
                pose.orientation += cccr.angularVelocity * dt;
                time += dt;

                cont = (gpr.path[gpr.path.size() - 1].position - targetPosition).norm() > reachpointError;
            }
            else
            {
                cont = false;

                if (!cccr.feasible)
                    gpr.pathState = PathState::unfeasible;
            }
            cont = cont && (time < maxTime);
        }

        if (time >= maxTime)
            gpr.pathState = PathState::timeout;

        gpr.finalError = (gpr.path[gpr.path.size() - 1].position - targetPosition).norm();

        return gpr;
    }

    double curveLength(vector<RobotPose> posePath)
    {
        double length = 0;
        for (int i = 0; i < posePath.size() - 1; i++)
            length += (posePath[i + 1].position - posePath[i].position).norm();

        return length;
    }

    GenerateManyPathsResult CBFCircPlanMany(RobotPose startingPose, VectorXd targetPosition, MapQuerier querier, double maxTime,
                                            double reachpointError, double deltaTime, Parameters param)
    {
        GenerateManyPathsResult gmpr;

        gmpr.pathOmega = {posRotZ(), -posRotZ(), Matrix3d::Zero()};
        gmpr.pathResults = {};
        gmpr.atLeastOnePathReached = false;

        double maxTimeTemp = maxTime;

        for (int i = 0; i < gmpr.pathOmega.size(); i++)
        {
            GeneratePathResult gpr = CBFCircPlanOne(startingPose, targetPosition, querier, gmpr.pathOmega[i], maxTimeTemp, reachpointError, deltaTime, param);
            double distToGoal = (startingPose.position - targetPosition).norm();
            gmpr.pathResults.push_back(gpr);
            gmpr.pathLenghts.push_back((gpr.pathState == PathState::sucess) ? curveLength(gpr.path) : VERYBIGNUMBER + distToGoal);
            gmpr.atLeastOnePathReached = gmpr.atLeastOnePathReached || (gpr.pathState == PathState::sucess);

            if ((gpr.pathState == PathState::sucess) &&
                (gmpr.pathLenghts[i] <= param.acceptableRationPlanning * distToGoal))
            {
                maxTimeTemp = 0.2;
            }
        }

        vector<int> ind = sortGiveIndex(gmpr.pathLenghts);

        gmpr.bestPathSize = gmpr.pathLenghts[ind[0]];
        gmpr.bestOmega = gmpr.pathOmega[ind[0]];
        gmpr.bestPath = gmpr.pathResults[ind[0]];

        return gmpr;
    }

    RadialDistanceResult computeDistRadial(vector<VectorXd> points, VectorXd position, double smoothingParam)
    {
        RadialDistanceResult rdr;
        vector<double> halfSqDistance = {};
        vector<VectorXd> distanceVector = {};

        for (int i = 0; i < points.size(); i++)
        {
            // Transform point
            halfSqDistance.push_back(0.5 * (points[i] - position).squaredNorm());
            distanceVector.push_back(points[i] - position);
        }

        SoftSelectMinResult ssmrHalfSqDistance = softSelectMin(3, halfSqDistance, distanceVector, smoothingParam);
        rdr.halfSqDistance = ssmrHalfSqDistance.softMin;
        rdr.gradDistance = ssmrHalfSqDistance.selected;

        return rdr;
    }

    VectorXd correctPoint(VectorXd point, vector<VectorXd> neighborPoints, Parameters param)
    {

        int k = 0;
        VectorXd pointCorrected = point;
        RadialDistanceResult rdr = computeDistRadial(neighborPoints, pointCorrected, param.smoothingParam);
        double currentDist, newDist;

        do
        {
            currentDist = rdr.halfSqDistance;
            pointCorrected += param.stepCorrectPoint * rdr.gradDistance.normalized();
            rdr = computeDistRadial(neighborPoints, pointCorrected, param.smoothingParam);
            newDist = rdr.halfSqDistance;
            k++;
        } while (newDist > currentDist && (k < param.noMaxIterationsCorrectPoint));

        return pointCorrected;
    }

    bool pathFree(vector<RobotPose> path, MapQuerier querier, int initialIndex, int finalIndex, Parameters param)
    {
        if (finalIndex - initialIndex < 5)
            return true;
        else
        {
            int midIndex = (int)(finalIndex + initialIndex) / 2;
            if (computeDist(querier(path[midIndex].position, param.sensingRadius), path[midIndex], param).distance < param.distPathFree)
                return false;
            else
                return pathFree(path, querier, initialIndex, midIndex, param) && pathFree(path, querier, midIndex, finalIndex, param);
        }
    }

    vector<RobotPose> generateSimplePath(vector<RobotPose> originalPath, MapQuerier querier, int initialIndex, int finalIndex, Parameters param)
    {
        RobotPose startingPose = originalPath[initialIndex];
        RobotPose endingPose = originalPath[finalIndex];
        vector<RobotPose> simplePath = {};

        int N = param.generateSimplePathDiv;
        for (int i = 0; i < N; i++)
        {
            RobotPose intPose;
            double fat = ((double)i) / ((double)(N - 1));
            intPose.position = startingPose.position + (endingPose.position - startingPose.position) * fat;
            intPose.orientation = startingPose.orientation + (endingPose.orientation - startingPose.orientation) * fat;
            simplePath.push_back(intPose);
        }

        if (pathFree(simplePath, querier, 0, simplePath.size() - 1, param))
            return simplePath;
        else
            return {};
    }

    vector<RobotPose> correctPath(vector<RobotPose> originalPath, MapQuerier querier, Parameters param)
    {
        vector<double> pathLength = {0};
        vector<RobotPose> modifiedPath = {originalPath[0]};

        for (int i = 0; i < originalPath.size() - 1; i++)
        {
            double delta = (originalPath[i + 1].position - originalPath[i].position).norm() +
                           abs(originalPath[i + 1].orientation - originalPath[i].orientation);
            pathLength.push_back(pathLength[i] + delta);
            modifiedPath.push_back(originalPath[i + 1]);
        }
        for (int i = 0; i < pathLength.size(); i++)
            pathLength[i] = pathLength[i] / pathLength[pathLength.size() - 1];

        for (int i = 0; i < originalPath.size(); i++)
        {
            double fat = 16 * pow(pathLength[i] * (1 - pathLength[i]), 2);

            for (int j = 0; j < param.noIterationsCorrectPath; j++)
            {
                DistanceResult dr = computeDist(querier(modifiedPath[i].position, param.sensingRadius), modifiedPath[i], param);

                fat = sqrt(1.0 + VERYSMALLNUMBER - pathLength[i]) * max(0.0, 1.0 - dr.distance / param.distCutoffCorrect);

                double norm = sqrt(dr.gradSafetyPosition.squaredNorm() + pow(dr.gradSafetyOrientation, 2)) + VERYSMALLNUMBER;
                modifiedPath[i].position += fat * param.correctPathStep * dr.gradSafetyPosition / norm;
                modifiedPath[i].orientation += fat * param.correctPathStep * dr.gradSafetyOrientation / norm;
            }
        }

        return modifiedPath;
    }

    vector<RobotPose> optimizePath(vector<RobotPose> originalPath, MapQuerier querier, Parameters param)
    {

        vector<RobotPose> path = {};

        // Reduze size
        int indexPath = -1;
        int currentIndex = originalPath.size() - 1;
        int i = 0;

        while (indexPath != originalPath.size() - 1 && i < param.noMaxOptimizePath)
        {
            vector<RobotPose> tryPath = generateSimplePath(originalPath, querier, 0, currentIndex, param);
            if (tryPath.size() > 0)
            {
                path = tryPath;
                indexPath = currentIndex;
                currentIndex = (originalPath.size() - 1 + currentIndex) / 2;
            }
            else
                currentIndex = currentIndex / 2;

            i++;
        }

        vector<RobotPose> optimizedPath = {};

        for (int i = 0; i < path.size(); i++)
            optimizedPath.push_back(path[i]);

        for (int i = indexPath + 1; i < originalPath.size(); i++)
            optimizedPath.push_back(originalPath[i]);

        // Correct path
        optimizedPath = correctPath(optimizedPath, querier, param);

        // Upsample:
        optimizedPath = upsample(optimizedPath, param.upsampleMinPos, param.upsampleMinOri);

        // Filter
        vector<RobotPose> finalPath = {};

        for (int i = 0; i < optimizedPath.size(); i++)
        {
            VectorXd pos = VectorXd::Zero(3);
            double ori = 0;
            int lasti = optimizedPath.size()-1;
            int jmin = i < param.filterWindow ? 0 : i - param.filterWindow;
            int jmax = i > lasti-param.filterWindow? lasti: i+param.filterWindow;
            double N = (double)(jmax - jmin) + 1;
            for (int j = jmin; j <= jmax; j++)
            {
                pos += optimizedPath[j].position;
                ori += optimizedPath[j].orientation;
            }
            pos = pos / N;
            ori = ori / N;

            RobotPose pose;
            pose.position = pos;
            pose.orientation = ori;
            finalPath.push_back(pose);
        }

        return finalPath;
    }

    vector<RobotPose> upsample(vector<RobotPose> path, double minDistPos, double minDistOri)
    {
        vector<RobotPose> upsampledPath = {};

        for (int i = 0; i < path.size() - 1; i++)
        {
            VectorXd deltaPos = path[i + 1].position - path[i].position;
            double deltaOri = path[i + 1].orientation - path[i].orientation;
            int N = (int)max(deltaPos.norm() / minDistPos, abs(deltaOri) / minDistOri);

            RobotPose newPose = path[i];
            for (int j = 0; j <= N; j++)
            {
                upsampledPath.push_back(newPose);
                newPose.position += deltaPos / ((double)N);
                newPose.orientation += deltaOri / ((double)N);
            }
        }

        upsampledPath.push_back(path[path.size() - 1]);

        return upsampledPath;
    }

    VectorFieldResult vectorField(RobotPose pose, vector<RobotPose> path, Parameters param)
    {

        // Find the closest point in the curve
        double dmin = VERYBIGNUMBER;
        double dminTemp;
        int ind = 0;

        for (int i = 0; i < path.size(); i++)
        {
            dminTemp = sqrt(0.5 * (path[i].position - pose.position).squaredNorm() + (1.0 - cos(path[i].orientation - pose.orientation)));
            if (dminTemp < dmin)
            {
                dmin = dminTemp;
                ind = i;
            }
        }

        VectorFieldResult vfr;
        vfr.distance = dmin;
        vfr.index = ind;

        VectorXd pi = path[ind].position;
        double thetai = path[ind].orientation;
        VectorXd gradD = vec3d(pi[0] - pose.position[0], pi[1] - pose.position[1], sin(thetai - pose.orientation));

        // Compute the normal vector
        VectorXd N = gradD / (gradD.norm() + VERYSMALLNUMBER);

        // Compute the tangent vector
        VectorXd T = VectorXd::Zero(3);

        if (ind == 0)
        {
            double dcos = cos(path[1].orientation) - cos(path[0].orientation);
            double dsin = sin(path[1].orientation) - sin(path[0].orientation);
            double dtheta = -sin(path[0].orientation) * dcos + cos(path[0].orientation) * dsin;
            T = vec3d(path[1].position[0] - path[0].position[0], path[1].position[1] - path[0].position[1], dtheta);
        }
        else
        {
            double dcos = cos(path[ind].orientation) - cos(path[ind - 1].orientation);
            double dsin = sin(path[ind].orientation) - sin(path[ind - 1].orientation);
            double dtheta = -sin(path[ind].orientation) * dcos + cos(path[ind].orientation) * dsin;
            T = vec3d(path[ind].position[0] - path[ind - 1].position[0], path[ind].position[1] - path[ind - 1].position[1], dtheta);
        }

        T = T / (T.norm() + VERYSMALLNUMBER);

        // Compute the G and H gains
        double G = (2 / M_PI) * atan(param.vectorFieldAlpha * sqrt(dmin));
        double H = sqrt(1 - (1 - VERYSMALLNUMBER) * G * G);

        // Compute the final vector field:
        VectorXd v = param.maxTotalVel * (0.5 * G * N + H * T);

        vfr.linearVelocity = vec3d(v[0], v[1], 0);
        vfr.angularVelocity = v[2];


        return vfr;
    }
    CBFControllerResult CBFController(RobotPose pose, VectorXd targetLinearVelocity, double targetAngularVelocity,
                                      vector<VectorXd> neighborPoints, Parameters param)
    {
        DistanceResult dr = computeDist(neighborPoints, pose, param);
        VectorXd ud = Vector3d::Zero(3);
        ud << targetLinearVelocity[0], targetLinearVelocity[1], targetAngularVelocity;

        MatrixXd H = 2 * Matrix3d::Identity(3, 3);
        VectorXd f = -2 * ud;
        MatrixXd A = MatrixXd::Zero(1, 3);
        A << dr.gradSafetyPosition[0], dr.gradSafetyPosition[1], dr.gradSafetyOrientation;
        VectorXd b = VectorXd::Zero(1);

        double cbfConst;
        if (dr.distance - param.distanceMarginPlan > 0)
            cbfConst = -param.alphaCBFPositive * (dr.distance - param.distanceMarginLowLevel);
        else
            cbfConst = -param.alphaCBFNegative * (dr.distance - param.distanceMarginLowLevel);

        b << cbfConst;

        VectorXd u = solveQP(H, f, A, b);

        CBFControllerResult ccr;
        ccr.distanceResult = dr;
        ccr.linearVelocity = VectorXd::Zero(3);

        if (u.rows() > 0)
        {
            if (u.norm() >= param.maxTotalVel)
                u = param.maxTotalVel * u / (u.norm());

            ccr.linearVelocity << u[0], u[1], 0;
            ccr.angularVelocity = u[2];
            ccr.feasible = true;
        }
        else
        {
            ccr.angularVelocity = 0;
            ccr.feasible = false;
        }

        return ccr;
    }
}
