#pragma once
#include <functional>
#include <vector>
#include <type_traits>
#include <cassert>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include "parallel_types.hpp"

namespace parallel {
/** Wrapper around MPI C bindings.
 * Implements operations useful to MC codes with special attention to templating
 */
class Mpi {
private:
/** Rank of root process.
 */
    static constexpr int kRoot = 0;
/** Size of FAN-IN of HeirarchyVectorReduce 
 */
    static constexpr int kVectorHeirarchyBase = 4;
/** Size of FAN-IN of HeirarchReduce 
 */
    static constexpr int kScalarHeirarchyBase = 20;
    int world_rank_;
    int world_size_;
    int tag_;
    Heirarchy vector_heirarchy;
    Heirarchy scalar_heirarchy;

/** Gives a tag for synchronizing asynchronous collective operations.
 */
    int GetTag();
/** Mechanism for synchronizing tags.
 */
    void IncrTag(int count);
/** Gets rank of current process on communicator comm.
 */
    int rank(MPI_Comm comm);
/** Gets size of communicator comm.
 */
    int size(MPI_Comm comm);
/** Returns true if current process is the root process on the communicator (rank = kRoot).
 */
    bool is_root(MPI_Comm comm);
/** Executes target on root rank.
 */
    void ExecRoot(std::function<void()> target, MPI_Comm comm);
/** Applies reduce over the vector of all values in communicator on root process.
 * Returns default constructed value for non-root processes
 */
    template<typename T> auto Reduce(const T& value, std::function<T(std::vector<T>&)> reduce, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;
/** Applies reduce over the vector of all values in communicator on all processes.
 */
    template<typename T> auto ReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;
/** Vector version of Reduce.
 * The size of value is not required to be the same across processes.
 * Probes on the root rank for an evelope and allocates a buffer accordingly.
 * Applies reduce on the received vectors with an accumulator.
 * Returns default constructed value for non-root processes.
 */
    template<typename T> auto VectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>& accumulator, const std::vector<T>& value)> reduce, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;
/** Sends/Receives variable length data in Packets to the specified destinations.
 * Uses an intial Alltoall to send data length then
 * allocates a buffer and intiates an Alltoallv.
 * Returns raw buffer and displacments, respectively.
 */
    template<typename T> auto SparseGather(const std::vector<Packet<T>>& packets) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::pair<std::vector<T>, std::vector<int>>>;
/** Pass through MPI_Gather.
 */
    template<typename T> auto Gather(const T& value, MPI_Comm comm) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;
/** Pass through MPI_Allgather.
 */
    template<typename T> auto AllGather(const T& value, MPI_Comm comm) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;
/** Pass through MPI_Send.
 */
    template<typename T> auto Send(const T& value, int target_rank, MPI_Comm comm) ->  
    std::enable_if_t<std::is_trivially_copyable<T>::value, void>;
/** Pass through MPI_Send for vectors.
 */
    template<typename T> auto Send(const T& value, int target_rank, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, void>>;
/** Asynchronous send operation.
 * Returns a AsyncOp containing the send buffer for the operation.
 */
    template<typename T> auto SendAsync(const T& value, int target_rank, MPI_Comm comm) ->  
    std::enable_if_t<std::is_trivially_copyable<T>::value, AsyncOp<T>>;
/** Asynchronous send operation for vectors.
 */
    template<typename T> auto SendAsync(const T& value, int target_rank, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, AsyncOp<typename T::value_type>>>;
/** Asynchronous send operation for vectors where value will not be used again.
 */
    template<typename T> auto SendAsync(std::vector<T>* value, int target_rank, MPI_Comm comm) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, AsyncOp<T>>;
/** Pass through of MPI_Recv.
 */
    template<typename T> auto Receive(int source_rank, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;
/** Pass through of MPI_Recv for vectors.
 * Probes for the message size and allocates a buffer accordingly.
 */
    template<typename T> auto Receive(int source_rank, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, T>>;
/** Pass through of MPI_Barrier.
 */
    void Barrier(MPI_Comm comm);
    
public:
/** Initializes MPI if not already and constructs the hairarchial reduction trees.
 */
    Mpi();
/** Finalizes MPI if not already.
 * Note: When the first instance of this class is destroyed further communication is not possible.
 * This should change in the future.
 */
    ~Mpi();
/** Process rank on MPI_COMM_WORLD.
 */
    int rank();
/** Size of MPI_COMM_WORLD.
 */
    int size();
/** Pass through of is_root with MPI_COMM_WORLD.
 */
    bool is_root();
/** Pass through of ExecRoot with MPI_COMM_WORLD.
 */
    void ExecRoot(std::function<void()> target) { ExecRoot(target, MPI_COMM_WORLD); }
/** Scalar reduction operation operating on a tree.
 * Uses tree constructed earlier to reduce in a series of Reduce operations on the small communicators.
 * Results in O(logN) complexity.
 * Returns default constructed return type on the non-root processes
 */
    template<typename T> auto HeirarchyReduce(const T& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;
/** Gives a SparseGather'd vector without rank information.
 * Discards the displacement information given by SparseGather.
 */
    template<typename T> auto SparseScalarGather(const std::vector<Packet<T>>& packets) -> std::vector<T>;
/** Splits data returned by SparseGather up into Packets.
 * SparseScalarGather should be preferred where possible as this method has the additional overhead of splitting up the buffer.
 */
    template<typename T> auto SparseVectorGather(const std::vector<Packet<T>>& packets) -> std::vector<Packet<T>>;
/** Applies Reduce on MPI_COMM_WORLD
 */
    template<typename T> auto Reduce(const T& value, std::function<T(std::vector<T>&)> reduce) { return Reduce<T>(value, reduce, MPI_COMM_WORLD); }
/** Applies ReduceeToAll on MPI_COMM_WORLD
 */
    template<typename T> auto ReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce) { return ReduceToAll<T>(value, reduce, MPI_COMM_WORLD); }
/** Returns a value for HeirarchyReduce on all processes.
 * Applies HeirarchyReduce and broadcasts the result.
 */
    template<typename T> auto HeirarchyReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;
/** Applies VectorReduce on MPI_COMM_WORLD
 */
    template<typename T> auto VectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>& accumulator, const std::vector<T>& value)> reduce) { return VectorReduce<T>(value, reduce, MPI_COMM_WORLD); }
/** Applies Gather on MPI_COMM_WORLD
 */
    template<typename T> auto Gather(const T& value) { return Gather<T>(value, MPI_COMM_WORLD); }
/** Applies AllGather on MPI_COMM_WORLD
 */
    template<typename T> auto AllGather(const T& value) { return AllGather<T>(value, MPI_COMM_WORLD); }
/** Applies Send on MPI_COMM_WORLD
 */
    template<typename T> auto Send(const T& value, int target_rank) { return Send<T>(value, target_rank, MPI_COMM_WORLD); }
/** Applies SendAsync on MPI_COMM_WORLD
 */
    template<typename T> auto SendAsync(const T& value, int target_rank) { return SendAsync<T>(value, target_rank, MPI_COMM_WORLD); }
/** Applies SendAsync on MPI_COMM_WORLD
 */
    template<typename T> auto SendAsync(std::vector<T>* value, int target_rank) { return SendAsync<T>(value, target_rank, MPI_COMM_WORLD); }
/** Applies Receive on MPI_COMM_WORLD
 */
    template<typename T> auto Receive(int source_rank) { return Receive<T>(source_rank, MPI_COMM_WORLD); }
/** Vector reduction operation operating on a tree.
 * Uses tree constructed earlier to reduce in a series of VectorReduce operations on the small communicators.
 * Results in O(logN) complexity.
 * Returns default constructed return type on the non-root processes.
 */
    template<typename T> auto HeirarchyVectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>& accumulator, const std::vector<T>& value)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;
/** Applies Broadcast on MPI_COMM_WORLD.
 */
    template<typename T> auto Broadcast(const T& value) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;
/** Applies Barrier on MPI_COMM_WOLRD.
 */
    void Barrier();
};


template<typename T> auto Mpi::Reduce(const T& value, std::function<T(std::vector<T>&)> reduce, MPI_Comm comm) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, T> {
    std::vector<T> data_buffer;
    if(is_root(comm)) {
        data_buffer.resize(size(comm));
    }

    MPI_Gather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, kRoot, comm);

    if(is_root(comm)) {
        return reduce(data_buffer);
    }else {
        return { };
    }
}

template<typename T> auto Mpi::HeirarchyReduce(const T& value, std::function<T(std::vector<T>&)> reduce) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, T> {
    T data = value;
    for(auto comm : scalar_heirarchy.comms()) {
        data = Reduce(data, reduce, comm);
    }
    if(is_root(MPI_COMM_WORLD)) {
        return data;
    }else {
        return { };
    }
}

template<typename T> auto Mpi::SparseScalarGather(const std::vector<Packet<T>>& packets) -> std::vector<T> {
    // discard displacement data
    return SparseGather(packets).first;
}

template<typename T> auto Mpi::SparseVectorGather(const std::vector<Packet<T>>& packets) -> std::vector<Packet<T>> {
    auto data = SparseGather(packets);
    std::vector<Packet<T>> out_packets;
    // use displacement data to load packet vector
    for(int k = 0; k < data.second.size(); ++k) {
        int size = (k+1 != data.second.size() ? data.second[k+1] : data.first.size()) - data.second[k];
        if(size > 0) {
            out_packets.emplace_back();
            out_packets.back().rank = k;
            int displacement = data.second[k];
            out_packets.back().data.insert(out_packets.back().data.begin(), data.first.begin() + displacement, data.first.begin() + displacement + size);
        }
    }
    return out_packets;
}

template<typename T> auto Mpi::SparseGather(const std::vector<Packet<T>>& packets) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, std::pair<std::vector<T>, std::vector<int>>> {
    // Get data count
    std::vector<int> send_counts(size(), 0);
    for(auto& p : packets) {
        send_counts.at(p.rank) = p.data.size();
    }
    std::vector<int> recv_counts(size());
    MPI_Request count_atoa_request;
    MPI_Ialltoall(send_counts.data(), sizeof(int), MPI_BYTE, recv_counts.data(), sizeof(int), MPI_BYTE, MPI_COMM_WORLD, &count_atoa_request);
    
    // calculate send displacements, load send buffer
    std::vector<int> send_displacements(size(), 0);
    std::vector<T> send_data;
    send_data.reserve(std::accumulate(send_counts.begin(), send_counts.end(), 0));
    for(auto& p : packets) {
        send_displacements[p.rank] = send_data.size() * sizeof(T);
        send_data.insert(send_data.end(), p.data.begin(), p.data.end());
    }
    std::transform(send_counts.begin(), send_counts.end(), send_counts.begin(), [&](int v) {return v * sizeof(T);});

    MPI_Wait(&count_atoa_request, MPI_STATUS_IGNORE);

   // Calculate receive offsets
    std::vector<int> recv_displacements(size());
    int partial_sum = 0;
    std::transform(recv_counts.begin(), recv_counts.end(), recv_displacements.begin(), [&](int v) -> int {
        int displ = partial_sum;
        partial_sum += v;
        return displ;
    });
    std::vector<T> recv_data(partial_sum);

    std::vector<int> recv_byte_displacements(size());
    std::transform(recv_displacements.begin(), recv_displacements.end(), recv_byte_displacements.begin(), [&](int v) {return v * sizeof(T);});
    std::transform(recv_counts.begin(), recv_counts.end(), recv_counts.begin(), [&](int v) {return v * sizeof(T);});
    // send data
    MPI_Alltoallv(send_data.data(), send_counts.data(), send_displacements.data(), MPI_BYTE, recv_data.data(), recv_counts.data(), recv_byte_displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
    return {recv_data, recv_displacements};
}

template<typename T> auto Mpi::ReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce, MPI_Comm comm) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, T> {
    std::vector<T> data_buffer(sizeof(T) * size(comm));

    MPI_Allgather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, comm);

    return reduce(data_buffer);
}

template<typename T> auto Mpi::HeirarchyReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, T> {
    T data = HeirarchyReduce(value, reduce);
    MPI_Bcast(&data, sizeof(T), MPI_BYTE, kRoot, MPI_COMM_WORLD);
    return data;
}

template<typename T> auto Mpi::VectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>&, const std::vector<T>&)> reduce, MPI_Comm comm) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>> {
    if(is_root(comm)) {
        MPI_Status status;
        MPI_Message message;
        int message_size;
        std::vector<T> accumulator;
        std::vector<std::vector<T>> data_buffers(size(comm));
        int tag = GetTag();
        std::vector<MPI_Request> requests(size(comm)-1);

        for(std::size_t k = 0; k < size(comm) - 1; ++k) {
            MPI_Mprobe(MPI_ANY_SOURCE, tag, comm, &message, &status);
            MPI_Get_count(&status, MPI_BYTE, &message_size);
            data_buffers[status.MPI_SOURCE].resize(message_size/sizeof(T));
            MPI_Imrecv(data_buffers[status.MPI_SOURCE].data(), message_size, MPI_BYTE, &message, &requests[k]);
        }

        std::vector<MPI_Status> finished_status(requests.size());
        std::vector<int> finished_index(requests.size());
        int finished_count;
        
        reduce(accumulator, value);
        std::size_t reduced_ranks = 1;
        do {
            MPI_Waitsome(requests.size(), requests.data(), &finished_count, finished_index.data(), finished_status.data());
            if(finished_count != MPI_UNDEFINED) {
                for(std::size_t k = 0; k < finished_count; ++k) {
                    int source_rank = finished_status[finished_index[k]].MPI_SOURCE;
                    reduce(accumulator, data_buffers[source_rank]);
                    data_buffers[source_rank].clear();
                }
            }
            reduced_ranks += finished_count;
        } while(reduced_ranks < size(comm));

        return accumulator;
    }else {
        MPI_Send(value.data(), value.size() * sizeof(T), MPI_BYTE, kRoot, GetTag(), comm);
        return { };
    }
}

template<typename T> auto Mpi::HeirarchyVectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>& accumulator, const std::vector<T>& value)> reduce) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>> {
    std::vector<T> data = value;
    for(auto comm : vector_heirarchy.comms()) {
        data = VectorReduce(data, reduce, comm);
    }
    IncrTag(vector_heirarchy.global_levels() - vector_heirarchy.local_levels());
    if(is_root(MPI_COMM_WORLD)) {
        return data;
    }else {
        return { };
    }
}

template<typename T> auto Mpi::Gather(const T& value, MPI_Comm comm) ->
std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>> {
        std::vector<T> data_buffer;
        if(is_root(comm)) {
            data_buffer.resize(size(comm));
        }
        MPI_Gather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, kRoot, comm);
        return data_buffer;
}

template<typename T> auto Mpi::AllGather(const T& value, MPI_Comm comm) ->
std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>> {
        std::vector<T> data_buffer(size(comm));
        MPI_Allgather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, comm);
        return data_buffer;
}

template<typename T> auto Mpi::Send(const T& value, int target_rank, MPI_Comm comm) ->  
std::enable_if_t<std::is_trivially_copyable<T>::value, void> {
    MPI_Send(&value, sizeof(T), MPI_BYTE, target_rank, 0, comm);
}

template<typename T> auto Mpi::Send(const T& value, int target_rank, MPI_Comm comm) -> 
std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, void>> {
    MPI_Send(value.data(), value.size() * sizeof(typename T::value_type), MPI_BYTE, target_rank, 0, comm);
}

template<typename T> auto Mpi::SendAsync(const T& value, int target_rank, MPI_Comm comm) ->  
std::enable_if_t<std::is_trivially_copyable<T>::value, AsyncOp<T>> {
    AsyncOp<T> op;
    op.buffer_ = std::vector<T>({value});
    MPI_ISend(op.buffer_.data(), sizeof(T), MPI_BYTE, target_rank, 0, comm, &op.request_);
    return op;
}

template<typename T> auto Mpi::SendAsync(const T& value, int target_rank, MPI_Comm comm) -> 
std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, AsyncOp<typename T::value_type>>> {
    AsyncOp<typename T::value_type> op;
    op.buffer_ = value;
    MPI_Isend(op.buffer_.data(), op.buffer_.size() * sizeof(typename T::value_type), MPI_BYTE, target_rank, 0, comm, &op.request_);
    return op;
}

template<typename T> auto Mpi::SendAsync(std::vector<T>* value, int target_rank, MPI_Comm comm) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, AsyncOp<T>> {
    AsyncOp<T> op;
    op.buffer_.swap(*value);
    MPI_Isend(op.buffer_.data(), op.buffer_.size() * sizeof(T), MPI_BYTE, target_rank, 0, comm, &op.request_);
    return op;
}

template<typename T> auto Mpi::Receive(int source_rank, MPI_Comm comm) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>> {
    T value;
    MPI_Recv(&value, sizeof(T), MPI_BYTE, source_rank, 0, comm, MPI_STATUS_IGNORE);
    return value;
}

template<typename T> auto Mpi::Receive(int source_rank, MPI_Comm comm) -> 
std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, T>> {
    MPI_Status status;
    MPI_Message message;
    int message_size;
    T data_buffer;
    
    MPI_Mprobe(source_rank, 0, comm, &message, &status);
    MPI_Get_count(&status, MPI_BYTE, &message_size);
    data_buffer.resize(message_size/sizeof(typename T::value_type));
    MPI_Mrecv(data_buffer.data(), message_size, MPI_BYTE, &message, MPI_STATUS_IGNORE);
    return data_buffer;
}

template<typename T> auto Mpi::Broadcast(const T& value) ->
std::enable_if_t<std::is_trivially_copyable<T>::value, T> {
    T data = value;
    MPI_Bcast(&data, sizeof(T), MPI_BYTE, kRoot, MPI_COMM_WORLD);
    return data;
}
}