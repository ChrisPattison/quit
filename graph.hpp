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
    bool IsConsistent() const; // Checks validity of the coefficient matrix
    int size() const; // Gets number of vertices
    int edges() const; // number of edges
    /*const*/ Eigen::SparseMatrix<EdgeType>& Adjacent() /*const*/;
    /*const*/ Eigen::Matrix<VertexType, Eigen::Dynamic, 1>& Fields() /*const*/;// TODO: make this better
};
