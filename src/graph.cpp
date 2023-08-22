#pragma once

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <list>
#include <math.h>
#include <vector>
#include <random>
#include <memory>
#include <chrono>


using namespace std;
using namespace Eigen;
using namespace std::chrono;

#include "graph.h"
#include "utils.h"
#include "plannerfunctions.h"

namespace CBFCirc
{

    GraphNode::GraphNode()
    {
        this->id = 0;
        this->position = VectorXd::Zero(3);
        this->outEdges = {};
        this->inEdges = {};
    };

    GraphEdge::GraphEdge()
    {
        this->id = 0;
        this->weight = 0;
        this->nodeIn = NULL;
        this->nodeOut = NULL;
        this->omega = Matrix3d::Zero();
    };

    Graph::Graph()
    {
        this->nodes = {};
        this->edges = {};
    }

    GraphNode *Graph::addNode(VectorXd position)
    {

        GraphNode *newNode = new GraphNode;
        newNode->id = nodes.size();
        newNode->position = position;
        nodes.push_back(newNode);

        GraphEdge *selfEdge = new GraphEdge;
        selfEdge->id = edges.size();
        selfEdge->nodeOut = newNode;
        selfEdge->nodeIn = newNode;
        selfEdge->omega = Matrix3d::Zero();
        selfEdge->weight = 0;
        selfEdge->forward = true;
        edges.push_back(selfEdge);

        return newNode;
    }

    void Graph::connect(GraphNode *nodeOut, GraphNode *nodeIn, double weight, Matrix3d omega)
    {
        // Create forward and reverse edges

        GraphEdge *newEdgeForward = new GraphEdge;
        newEdgeForward->id = edges.size();
        newEdgeForward->nodeOut = nodeOut;
        newEdgeForward->nodeIn = nodeIn;
        newEdgeForward->omega = omega;
        newEdgeForward->weight = weight;
        nodeOut->outEdges.push_back(newEdgeForward);
        nodeIn->inEdges.push_back(newEdgeForward);
        newEdgeForward->forward = true;

        edges.push_back(newEdgeForward);

        GraphEdge *newEdgeReverse = new GraphEdge;
        newEdgeReverse->id = edges.size();
        newEdgeReverse->nodeOut = nodeIn;
        newEdgeReverse->nodeIn = nodeOut;
        newEdgeReverse->omega = -omega;
        newEdgeReverse->weight = weight;
        nodeIn->outEdges.push_back(newEdgeReverse);
        nodeOut->inEdges.push_back(newEdgeReverse);
        newEdgeReverse->forward = false;

        edges.push_back(newEdgeReverse);
    }

    vector<GraphNode *> Graph::getNeighborNodes(VectorXd position, double radius)
    {

        vector<GraphNode *> nNodes;
        for (int i = 0; i < nodes.size(); i++)
            if ((position - nodes[i]->position).norm() <= radius)
                nNodes.push_back(nodes[i]);

        return nNodes;
    }

    GraphNode *Graph::getNearestNode(VectorXd position)
    {
        return getNearestNodeList(position)[0];
    }

    vector<GraphNode *> Graph::getNearestNodeList(VectorXd position)
    {
        vector<double> distUnsorted;
        for (int i = 0; i < nodes.size(); i++)
            distUnsorted.push_back((position - nodes[i]->position).norm());

        vector<int> ind = sortGiveIndex(distUnsorted);

        vector<GraphNode *> nodesSorted;
        for (int i = 0; i < ind.size(); i++)
            nodesSorted.push_back(nodes[ind[i]]);

        return nodesSorted;
    }

    double computeDistPath(vector<GraphEdge *> edges)
    {
        double d = 0;
        for (int i = 0; i < edges.size(); i++)
            d += edges[i]->weight;

        return d;
    }

    NewExplorationPointResult Graph::getNewExplorationPoint(RobotPose pose, MapQuerier querier, vector<vector<VectorXd>> frontier, Parameters param)
    {

        NewExplorationPointResult sntr;
        vector<double> valueUnsorted = {};
        vector<VectorXd> pointUnsorted = {};
        vector<int> indexGraphUnsorted = {};
        vector<Matrix3d> omegaUnsorted = {};

        // Get closest node that can be reached
        vector<GraphNode *> closestNodesToCurrent = getNearestNodeList(pose.position);
        GraphNode *closestNodeToPosition;
        bool found = false;
        bool triedAll = false;
        int k = 0;

        cout << "Start planning. Trying to find entry point.";
        auto start = high_resolution_clock::now();
        do
        {
            closestNodeToPosition = closestNodesToCurrent[k];
            k++;
            found = CBFCircPlanMany(pose, closestNodeToPosition->position, querier, param.maxTimeSampleExploration,
                                    param.plannerReachError, param.deltaTimeSampleExploration, param)
                        .atLeastOnePathReached;

            triedAll = k >= closestNodesToCurrent.size();
        } while (!found && !triedAll);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Finished in "<<((double) duration.count()/1000000.0)<<" seconds!";

        if (!found)
        {
            cout << "Failed to go to ANY point in the graph!";
            sntr.success = false;
        }
        else
        {
            cout << "Success to go to a point of the graph ("<<k<<" out of "<<closestNodesToCurrent.size()<<")";
            sntr.success = true;
        }
            

        cout << "Starting frontier exploration (Type A)..." << frontier.size() << " clusters found";

        if (sntr.success)
        {
            for (int i = 0; i < frontier.size(); i++)
            {
                cout << "Computing for frontier point: "<<i;
                double bestDist = -VERYBIGNUMBER;
                double tempDist;
                VectorXd bestPoint = VectorXd::Zero(3);
                for (int j = 0; j < frontier[i].size(); j++)
                {
                    tempDist = computeDistRadial(querier(frontier[i][j], param.sensingRadius), frontier[i][j], param.smoothingParam).halfSqDistance;
                    if ((tempDist > 0.5 * pow(param.distanceMinBeta, 2)) && (tempDist > bestDist))
                    {
                        bestDist = tempDist;
                        bestPoint = frontier[i][j];
                    }
                }

                if (bestDist > -VERYBIGNUMBER / 2)
                {
                    vector<GraphNode *> closestNodesToBestPoint = getNearestNodeList(bestPoint);

                    GraphNode *bestNodeToExploration;
                    double bestValue = VERYBIGNUMBER;
                    Matrix3d bestOmega;
                    int jmax = (closestNodesToBestPoint.size() < param.noTriesClosestPoint) ? closestNodesToBestPoint.size() : param.noTriesClosestPoint;
                    bool cont = true;
                    int j =0;
                    while(cont)
                    {
                        GraphNode *nodeTry = closestNodesToBestPoint[j];
                        RobotPose poseTry;
                        poseTry.position = nodeTry->position;
                        poseTry.orientation = 0;

                        GenerateManyPathsResult gmpr1 = CBFCircPlanMany(poseTry, bestPoint, querier, param.maxTimeSampleExploration, param.plannerReachError,
                                                                        param.deltaTimeSampleExploration, param);
                        if (gmpr1.bestPathSize <= bestValue)
                        {
                            bestValue = gmpr1.bestPathSize;
                            bestOmega = gmpr1.bestOmega;
                            bestNodeToExploration = nodeTry;
                        }
                        j++;
                        cont = cont && (j<jmax);
                    }

                    if (bestValue < VERYBIGNUMBER / 2)
                    {
                        RobotPose poseBestPoint;
                        poseBestPoint.position = bestPoint;
                        poseBestPoint.orientation = 0;
                        GenerateManyPathsResult gmpr2 = CBFCircPlanMany(poseBestPoint, param.globalTargetPosition, querier,
                                                                        param.maxTimeSampleExploration, param.plannerReachError, 
                                                                        param.deltaTimeSampleExploration, param);

                        double dist1 = 100*(closestNodeToPosition->position - pose.position).norm();
                        double dist2 = 100*computeDistPath(getPath(closestNodeToPosition, bestNodeToExploration));
                        double dist3 = bestValue;
                        double dist4 = gmpr2.bestPathSize;

                        pointUnsorted.push_back(bestPoint);
                        valueUnsorted.push_back(dist1 + dist2 + dist3 + dist4);
                        indexGraphUnsorted.push_back(bestNodeToExploration->id);
                        omegaUnsorted.push_back(bestOmega);
                    }
                }
            }

            sntr.value = {};
            sntr.points = {};
            sntr.index = {};

            if (valueUnsorted.size() > 0)
            {
                vector<int> ind = sortGiveIndex(valueUnsorted);
                for (int i = 0; i < ind.size(); i++)
                {
                    sntr.value.push_back(valueUnsorted[ind[i]]);
                    sntr.points.push_back(pointUnsorted[ind[i]]);
                    sntr.index.push_back(indexGraphUnsorted[ind[i]]);
                }


                //sntr.indexClosestNode = sntr.index[0];
                sntr.bestExplorationPosition = sntr.points[0];
                sntr.pathToExplorationPoint = getPath(closestNodeToPosition, this->nodes[sntr.index[0]]);
                sntr.bestOmega = omegaUnsorted[sntr.index[0]];
                sntr.success = true;


                cout << "SUCCESS: Point found!";
            }
            else
            {
                cout << "ERROR: No point found using Type A algorithm...";
                sntr.success = false;
            }
        }

        return sntr;
    }

    vector<GraphEdge *> Graph::getPath(GraphNode *origin, GraphNode *target)
    {
        vector<double> V;
        vector<int> edgeToTake;

        for (int i = 0; i < nodes.size(); i++)
        {
            if (i == target->id)
                V.push_back(0.0);
            else
                V.push_back(VERYBIGNUMBER);
            edgeToTake.push_back(-1);
        }

        vector<double> Vold;

        for (int i = 0; i < V.size(); i++)
            Vold.push_back(V[i]);

        bool cont = true;

        do
        {
            for (int i = 0; i < nodes.size(); i++)
            {
                for (int j = 0; j < nodes[i]->outEdges.size(); j++)
                {
                    int idOut = nodes[i]->outEdges[j]->nodeIn->id;
                    double w = nodes[i]->outEdges[j]->weight;

                    if (w + Vold[idOut] < V[i])
                    {
                        V[i] = w + Vold[idOut];
                        edgeToTake[i] = nodes[i]->outEdges[j]->id;
                    }
                }
            }

            cont = false;
            for (int i = 0; i < Vold.size(); i++)
            {
                cont = cont || (V[i] < Vold[i]);
                Vold[i] = V[i];
            }

        } while (cont);

        vector<GraphEdge *> path = {};

        if (V[origin->id] < VERYBIGNUMBER)
        {
            // Its possible!

            // Add self edge

            for (int i = 0; i < edges.size(); i++)
                if ((edges[i]->nodeIn->id == origin->id) && (edges[i]->nodeOut->id == origin->id))
                    path.push_back(edges[i]);

            int currentNode = origin->id;
            int currentEdgeIndex;

            while (currentNode != target->id)
            {
                currentEdgeIndex = edgeToTake[currentNode];
                path.push_back(edges[currentEdgeIndex]);
                currentNode = edges[currentEdgeIndex]->nodeIn->id;
            }
        }

        return path;
    }
}
