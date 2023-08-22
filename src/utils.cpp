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

#include "utils.h"
#include "./quadprogpp/src/QuadProg++.hh"
#include "./quadprogpp/src/QuadProg++.cc"

namespace CBFCirc
{

    VectorXd solveQP(MatrixXd H, VectorXd f, MatrixXd A, VectorXd b, MatrixXd Aeq, VectorXd beq)
    {
        // Solve min_u (u'*H*u)/2 + f'*u
        // such that:
        // A*u >= b
        // Aeq*u = beq
        // The function assumes that H is a positive definite function (the problem is strictly convex)

        int n = H.rows();

        if (A.rows() == 0 && A.cols() == 0)
        {
            A = MatrixXd::Zero(0, n);
            b = VectorXd::Zero(0);
        }

        if (Aeq.rows() == 0 && Aeq.cols() == 0)
        {
            Aeq = MatrixXd::Zero(0, n);
            beq = VectorXd::Zero(0);
        }

        int meq = Aeq.rows();
        int mineq = A.rows();

        quadprogpp::Matrix<double> H_aux, Aeq_aux, A_aux;
        quadprogpp::Vector<double> f_aux, beq_aux, b_aux, u_aux;

        H_aux.resize(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                H_aux[i][j] = H(i, j);

        f_aux.resize(n);
        for (int i = 0; i < n; i++)
            f_aux[i] = f[i];

        Aeq_aux.resize(n, meq);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < meq; j++)
                Aeq_aux[i][j] = Aeq(j, i);

        beq_aux.resize(meq);
        for (int j = 0; j < meq; j++)
            beq_aux[j] = -beq[j];

        A_aux.resize(n, mineq);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < mineq; j++)
                A_aux[i][j] = A(j, i);

        b_aux.resize(mineq);
        for (int j = 0; j < mineq; j++)
            b_aux[j] = -b[j];

        u_aux.resize(n);

        double val = solve_quadprog(H_aux, f_aux, Aeq_aux, beq_aux, A_aux, b_aux, u_aux);

        if (val > 1.0E50)
        {
            // Problem is unfeasible
            VectorXd u(0);
            return u;
        }
        else
        {
            // Problem is feasible
            VectorXd u(n);

            for (int i = 0; i < n; i++)
                u[i] = u_aux[i];

            return u;
        }
    }

    MatrixXd matrixVertStack(MatrixXd A1, MatrixXd A2)
    {
        MatrixXd A(A1.rows() + A2.rows(), A1.cols());
        A << A1, A2;
        return A;
    }

    VectorXd vectorVertStack(VectorXd v1, VectorXd v2)
    {
        VectorXd v(v1.rows() + v2.rows());
        v << v1, v2;
        return v;
    }

    VectorXd vectorVertStack(double v1, VectorXd v2)
    {
        VectorXd v(1 + v2.rows());
        v << v1, v2;
        return v;
    }

    VectorXd vectorVertStack(VectorXd v1, double v2)
    {
        VectorXd v(v1.rows() + 1);
        v << v1, v2;
        return v;
    }

    VectorXd vectorVertStack(double v1, double v2)
    {
        VectorXd v(2);
        v << v1, v2;
        return v;
    }

    string printNumber(double x, int nochar)
    {
        double P1 = pow(10, nochar - 3);
        double P2 = 1 / P1;

        double y = P2 * round(x * P1);
        string str;
        if (x >= 0)
            str = " " + std::to_string(y).substr(0, nochar - 1);
        else
            str = std::to_string(y).substr(0, nochar - 1);

        while (str.size() < nochar)
            str += "0";

        return str;
    }

    string printVector(VectorXd v, int nochar)
    {
        string str = "[";
        for (int i = 0; i < v.rows() - 1; i++)
            str += printNumber(v[i], nochar) + ", ";

        str += printNumber(v[v.rows() - 1], nochar) + "]";

        return str;
    }

    string printVectorOctave(VectorXd v, int nochar)
    {
        string str = "[";
        for (int i = 0; i < v.rows() - 1; i++)
            str += printNumber(v[i], nochar) + " ";

        str += printNumber(v[v.rows() - 1], nochar) + "]";

        return str;
    }

    string printMatrix(MatrixXd M, int nochar)
    {
        string str = "";

        for (int i = 0; i < M.rows(); i++)
        {
            str += (i == 0) ? "[" : " ";
            for (int j = 0; j < M.cols(); j++)
                str += printNumber(M(i, j), nochar) + "  ";

            str += (i == M.rows() - 1) ? "]" : "\n";
        }

        return str;
    }

    double rand(double vMin, double vMax)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(vMin, vMax);

        return dist(mt);
    }

    VectorXd randVec(int n, double vMin, double vMax)
    {
        VectorXd v(n);

        for (int i = 0; i < n; i++)
        {
            v[i] = rand(vMin, vMax);
        }

        return v;
    }

    SoftSelectMinResult softSelectMin(int dim, vector<double> v, vector<VectorXd> vVec, double h)
    {
        double valMin = VERYBIGNUMBER;
        double sum = VERYSMALLNUMBER;
        double tempVal;
        VectorXd sumv = VectorXd::Zero(dim);

        for (int i = 0; i < v.size(); i++)
            valMin = min(valMin, v[i]);

        for (int i = 0; i < v.size(); i++)
        {
            tempVal = exp(-(v[i] - valMin) / h);
            sum += tempVal;
            sumv += tempVal * vVec[i];
        }

        SoftSelectMinResult ssmr;

        ssmr.selected = sumv / sum;
        ssmr.residue = -h * log(sum / (VERYSMALLNUMBER + (double)v.size()));
        ssmr.trueMin = valMin;
        ssmr.softMin = ssmr.trueMin + ssmr.residue;

        return ssmr;
    }

    vector<VectorXd> upsample(vector<VectorXd> points, double dist)
    {
        vector<VectorXd> upsampledPoints;
        VectorXd currentPoint;
        VectorXd nextPoint = points[0];
        VectorXd dir;
        double curDist;
        int jmax;

        for (int i = 1; i < points.size(); i++)
        {
            currentPoint = points[i - 1];
            nextPoint = points[i];
            curDist = (currentPoint - nextPoint).norm();

            jmax = (int)floor(curDist / dist) - 1;

            dir = (nextPoint - currentPoint) / (VERYSMALLNUMBER + curDist);

            for (int j = 0; j <= jmax; j++)
            {
                upsampledPoints.push_back(currentPoint + ((double)j) * dist * dir);
            }
        }
        upsampledPoints.push_back(nextPoint);

        return upsampledPoints;
    }

    // VectorXd correctPoint(VectorXd q, vector<VectorXd> (*querier)(VectorXd, double), double h, double sr, int N, double zmin, double zmax)
    // {

    //     int k = 0;
    //     VectorXd qc = q;
    //     SmoothDistanceResult smdr = smoothSignedDistance(qc, querier(q, sr), h, 0, 0, 0, zmin, zmax);
    //     double currentD, newD;

    //     do
    //     {
    //         currentD = smdr.D;
    //         qc += 0.1 * smdr.gradD;
    //         smdr = smoothSignedDistance(qc, querier(q, sr), h, 0, 0, 0, zmin, zmax);
    //         newD = smdr.D;
    //         k++;
    //     } while (newD > currentD && (k < N) && ((q - qc).norm() <= 0.25 * sr));

    //     return qc;
    // }

    bool checkLimitCycleorConvergence(vector<double> t, vector<VectorXd> point, double deltat, double deltad)
    {
        // Check for limit cycles or if moved only a little

        // With bisection it is probably faster... do it later

        for (int i = 0; i < point.size(); i++)
            for (int j = i + 1; j < point.size(); j++)
                if ((t[j] - t[i] >= deltat) && (point[j] - point[i]).norm() <= deltad)
                    return true;

        return false;
    }



    vector<int> sortGiveIndex(vector<double> v)
    {
        vector<int> idx(v.size());
        iota(idx.begin(), idx.end(), 0);

        stable_sort(idx.begin(), idx.end(),
                    [&v](size_t i1, size_t i2)
                    { return v[i1] < v[i2]; });

        return idx;
    }

    void printVectorsToCSV(ofstream *f, vector<VectorXd> points)
    {
        string str;

        for (int i = 0; i < points.size(); i++)
        {
            str = "";
            for (int j = 0; j < points[0].size(); j++)
                str += printNumber(points[i][j]) + ";";
            *f << str << std::endl;
        }
    }

    void printVectorsToCSV(ofstream *f, vector<double> points)
    {

        string str;

        for (int i = 0; i < points.size(); i++)
            str += printNumber(points[i]) + ";";

        *f << str << std::endl;
    }

    void printVectorVectorsToCSV(ofstream *f, vector<vector<VectorXd>> points, int rowNo)
    {
        string str;

        for (int i = 0; i < points.size(); i++)
        {
            for (int j = 0; j < points[i].size(); j++)
            {
                str = "";
                for (int k = 0; k < points[i][0].size(); k++)
                    str += printNumber(points[i][j][k]) + ";";

                *f << str << std::endl;
            }

            str = "";
            for (int k = 0; k < rowNo; k++)
                str += "nan;";

            *f << str << std::endl;
        }
    }

    bool checkPathFree(VectorXd qa, VectorXd qb, vector<VectorXd> (*querier)(VectorXd, double), double minDist, double lengthStop)
    {
        // Consider the line q(L) = qa*(1-L) + L*qb between (qa,qb). Returns q(L*), in which L* = max L, such that L <=1 and the line
        // between qa and q(L) is free

        if ((qa - qb).norm() <= lengthStop)
            return true;
        else
        {
            VectorXd qm = (qa + qb) / 2;
            if (querier(qm, minDist).size() > 0)
                return false;
            else
                return checkPathFree(qa, qm, querier, minDist, lengthStop) && checkPathFree(qm, qb, querier, minDist, lengthStop);
        }
    }

    vector<VectorXd> pointsFreeInRay(VectorXd qc, double R, double r, int N, vector<VectorXd> (*querier)(VectorXd, double), double minDist, double lengthStop)
    {
        vector<VectorXd> points = {};

        for (int i = 0; i < N; i++)
        {
            double theta = ((double)i) * (2 * PI) / ((double)N);
            VectorXd d = VectorXd::Zero(3);
            d << cos(theta), sin(theta), 0;
            if (checkPathFree(qc, qc + R * d, querier, minDist, lengthStop))
                points.push_back(qc + r * d);
        }

        return points;
    }

    Matrix3d posRotX()
    {
        Matrix3d OMEGA0 = Matrix3d::Zero();
        OMEGA0(1, 2) = -1;
        OMEGA0(2, 1) = 1;
        return OMEGA0;
    }

    Matrix3d posRotY()
    {
        Matrix3d OMEGA0 = Matrix3d::Zero();
        OMEGA0(0, 2) = 1;
        OMEGA0(2, 0) = -1;
        return OMEGA0;
    }

    Matrix3d posRotZ()
    {
        Matrix3d OMEGA0 = Matrix3d::Zero();
        OMEGA0(0, 1) = -1;
        OMEGA0(1, 0) = 1;
        return OMEGA0;
    }

    SafetyResult safetyCylinder(VectorXd point, double radius, double height)
    {
        SafetyResult sr;
        double h = 0.5;

        double a = point[1] * point[1] + point[2] * point[2] - radius * radius;
        double b = point[0] * point[0] - height * height / 4;
        double maxab = max(a, b);

        double A = exp((a - maxab) / (h * h));
        double B = exp((b - maxab) / (h * h));

        sr.safety = maxab + (h * h) * log((A + B) / 2);

        VectorXd va = VectorXd::Zero(3);
        va << 0, point[1], point[2];
        VectorXd vb = VectorXd::Zero(3);
        vb << point[0], 0, 0;

        sr.gradSafety = 2*(A * va + B * vb) / (A + B);



        return sr;
    }

    double signedDistCylinder(VectorXd point, double radius, double height)
    {
        double D = VERYBIGNUMBER;

        double r = sqrt(pow(point[1], 2) + pow(point[2], 2));
        double z = abs(point[0]);
        double insideD = max(r - radius, z - height / 2);
        if (insideD < 0)
            D = min(D, insideD);
        else
        {
            double outsideD = sqrt(pow(max(r - radius, 0.0), 2) + pow(max(z - height / 2, 0.0), 2));
            D = min(D, outsideD);
        }

        return D;
    }



    double computeMeanCurv(vector<VectorXd> q, int sampfac, int start, int end)
    {
        int endind = end < q.size() ? end : q.size();

        if (endind - start + 1 < 5 * sampfac)
            return 0;
        else
        {
            double meanCurv = 0;

            for (int i = (int)(start / sampfac) + 1; i < (int)(endind / sampfac) - 1; i++)
            {
                VectorXd T1 = (q[sampfac * i] - q[sampfac * (i - 1)]).normalized();
                VectorXd T2 = (q[sampfac * (i + 1)] - q[sampfac * i]).normalized();
                double ds = (q[sampfac * (i + 1)] - q[sampfac * (i - 1)]).norm();
                meanCurv += (T2 - T1).norm() / (VERYSMALLNUMBER + ds);
            }

            return meanCurv / ((double)((endind - start + 1) / sampfac));
        }
    }

    string getMatrixName(Matrix3d omega)
    {
        if(abs(omega(1, 0)-1) <= VERYSMALLNUMBER)
            return "PZ";
        if(abs(omega(1, 0)+1) <= VERYSMALLNUMBER)
            return "NZ";   
        if(abs(omega(1, 0)) + abs(omega(2, 1)) + abs(omega(0, 2)) <= VERYSMALLNUMBER)
            return "0";             
    }
    int getMatrixNumber(Matrix3d omega)
    {
        if(abs(omega(1, 0)-1) <= VERYSMALLNUMBER)
            return 2;
        if(abs(omega(1, 0)+1) <= VERYSMALLNUMBER)
            return -2;   
        if(abs(omega(1, 0)) + abs(omega(2, 1)) + abs(omega(0, 2)) <= VERYSMALLNUMBER)
            return 0;             
    }

    VectorXd vec3d(double x, double y, double z)
    {
        VectorXd vector = VectorXd::Zero(3);
        vector << x, y, z;
        return vector;
    }
}
