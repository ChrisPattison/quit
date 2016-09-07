#pragma once
#include "types.hpp"
#include "Eigen/Sparse"
#include "Eigen/Dense"

class Graph {
    Eigen::SparseMatrix<EdgeType> adjacent_;
    Eigen::Matrix<VertexType, Eigen::Dynamic, 1> fields_;
public:
    void Resize(int no_vertices, int no_couplers); // reserves memory for vertices and couplers
    void SetField(int vertex, EdgeType field); // Set the field on a vertex
    void AddEdge(int from, int to, EdgeType coupler); // Sets a non-zero coupler
    bool IsSymmetric() const; // Checks symmetry of the adjacency matrix (false implies bad time [TM])
    int Size() const; // Gets number of vertices
    /*const*/ Eigen::SparseMatrix<EdgeType>& Adjacent() /*const*/;
    /*const*/ Eigen::Matrix<VertexType, Eigen::Dynamic, 1>& Fields() /*const*/;// TODO: make this better
};
