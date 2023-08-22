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

#include "plannerfunctions.h"

namespace CBFCirc
{
    class GraphNode;
    class GraphEdge;

    class GraphNode
    {
    public:
        int id;
        VectorXd position;
        vector<GraphEdge *> outEdges;
        vector<GraphEdge *> inEdges;

        GraphNode();
    };

    class GraphEdge
    {
    public:
        int id;
        double weight;
        GraphNode *nodeOut;
        GraphNode *nodeIn;
        Matrix3d omega;
        bool forward;

        GraphEdge();
    };

    struct NewExplorationPointResult
    {
        vector<double> value;
        vector<VectorXd> points;
        vector<int> index;
        
        VectorXd bestExplorationPosition;
        Matrix3d bestOmega;
        vector<GraphEdge *> pathToExplorationPoint;
        bool success;
    };

    class Graph
    {
    public:
        vector<GraphNode *> nodes;
        vector<GraphEdge *> edges;

        Graph();
        GraphNode *addNode(VectorXd position);
        void connect(GraphNode *nodeOut, GraphNode *nodeIn, double weight, Matrix3d omega);
        vector<GraphNode *> getNeighborNodes(VectorXd position, double radius);
        GraphNode *getNearestNode(VectorXd position);
        vector<GraphNode *> getNearestNodeList(VectorXd position);
        NewExplorationPointResult getNewExplorationPoint(RobotPose pose, MapQuerier querier, vector<vector<VectorXd>> frontier, Parameters param);
        vector<GraphEdge *> getPath(GraphNode *origin, GraphNode *target);
    };

}
