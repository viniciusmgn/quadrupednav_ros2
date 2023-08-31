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

    const double VERYBIGNUMBER = 1000000;
    const double VERYSMALLNUMBER = 0.000001;
    const double PI = 3.14;
    const double E106 = 1000000;

    struct DataWithTimeStamp
    {
        double time;
        VectorXd data;
    };

    struct SmoothDistanceResult
    {
        double D;
        double trueD;
        VectorXd gradD;
    };

    struct SoftSelectMinResult
    {
        VectorXd selected;
        double softMin;
        double trueMin;
        double residue;
    };

    struct SafetyResult
    {
        double safety;
        VectorXd gradSafety;
    };

    VectorXd solveQP(MatrixXd H, VectorXd f, MatrixXd A = MatrixXd(0, 0), VectorXd b = VectorXd(0), MatrixXd Aeq = MatrixXd(0, 0), VectorXd beq = VectorXd(0));
    MatrixXd matrixVertStack(MatrixXd A1, MatrixXd A2);
    VectorXd vectorVertStack(VectorXd v1, VectorXd v2);
    VectorXd vectorVertStack(double v1, VectorXd v2);
    VectorXd vectorVertStack(VectorXd v1, double v2);
    VectorXd vectorVertStack(double v1, double v2);
    string printNumber(double x, int nochar = 8);
    string printVector(VectorXd v, int nochar = 8);
    string printVectorOctave(VectorXd v, int nochar = 8);
    string printMatrix(MatrixXd M, int nochar = 8);
    double rand(double vMin, double vMax);
    VectorXd randVec(int n, double vMin, double vMax);
    SoftSelectMinResult softSelectMin(int dim, vector<double> v, vector<VectorXd> vVec, double h);
    vector<VectorXd> upsample(vector<VectorXd> points, double dist);
    bool checkLimitCycleorConvergence(vector<double> t, vector<VectorXd> point, double deltat, double deltad);
    vector<int> sortGiveIndex(vector<double> val);
    void printVectorsToCSV(ofstream *f, vector<VectorXd> points);
    void printVectorVectorsToCSV(ofstream *f, vector<vector<VectorXd>> points, int rowNo);
    void printVectorsToCSV(ofstream *f, vector<double> points);
    bool checkPathFree(VectorXd qa, VectorXd qb, vector<VectorXd> (*querier)(VectorXd, double), double minDist, double lengthStop);
    vector<VectorXd> pointsFreeInRay(VectorXd qc, double R, double r, int N, vector<VectorXd> (*querier)(VectorXd, double), double minDist, double lengthStop);
    Matrix3d posRotX();
    Matrix3d posRotY();
    Matrix3d posRotZ();
    SafetyResult safetyCylinder(VectorXd point, double radius, double height);
    double signedDistCylinder(VectorXd point, double radius, double height);
    double computeMeanCurv(vector<VectorXd> q, int sampfac, int start, int end);
    string getMatrixName(Matrix3d omega);
    int getMatrixNumber(Matrix3d omega);
    VectorXd vec3d(double x, double y, double z);
}
