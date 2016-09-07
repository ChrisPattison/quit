#include "graph.hpp"

void Graph::Resize(int no_vertices, int no_couplers) {
    fields_.resize(no_vertices);
    adjacent_.resize(no_vertices, no_vertices);
    adjacent_.reserve(no_couplers);
    fields_.setZero();
}

void Graph::SetField(int vertex, EdgeType field) {
    fields_(vertex) = field;
}

void Graph::AddEdge(int from, int to, EdgeType coupler) {
    adjacent_.insert(from, to) = coupler;
}

bool Graph::IsSymmetric() const {
    return adjacent_.isApprox(adjacent_.transpose());
}

Eigen::SparseMatrix<EdgeType>& Graph::Adjacent() {
    return adjacent_;
}

Eigen::Matrix<VertexType, Eigen::Dynamic, 1>& Graph::Fields() {
    return fields_;
}

int Graph::Size() const {
    return fields_.size();
}