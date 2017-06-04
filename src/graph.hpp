/* Copyright (c) 2016 C. Pattison
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
#pragma once
#include "types.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace propane {
/** Contains the fields and coefficients as well as useful methods for construction and error checking.
 * Contains an adjacency matrix of the problem where the nonzeros are the value of the coefficient J_ij between S_i and S_j.
 * The field h_i is that felt by S_i.
 */
class Graph {
    Eigen::SparseMatrix<EdgeType> adjacent_;
    Eigen::Matrix<EdgeType, Eigen::Dynamic, 1> fields_;
    bool field_nonzero_;
public:
/**  Resizes adjacency matrix and field vector.
 *  Zeros out field vector and reserves memory for adjacency matrix.
 */
    void Resize(int no_vertices, int no_couplers);
/** Sets the field on a particular vertex.
 */
    void SetField(int vertex, EdgeType field);
/** Sets a non-zero coefficient between S_from and S_to.
 * This is unidirecational and must be set in both directions in most cases.
 */
    void AddEdge(int from, int to, EdgeType coupler);
/** Checks validity of the adjacency matrix.
 * Checks for zero rows and asymmetry.
 */
    bool IsConsistent() const;
/** Gets number of vertices.
 */
    int size() const;
/** Number of edges with bidirectional asumption.
 */
    int edges() const;
/** Returns true if a field has been set.
 */
    bool has_field() const;
/** Reference to adjacency matrix.
 * This will be const in the future.
 */
    Eigen::SparseMatrix<EdgeType>& Adjacent();
/** Reference to field vector.
 * This will be const in the future.
 */
     Eigen::Matrix<EdgeType, Eigen::Dynamic, 1>& Fields();
};
}