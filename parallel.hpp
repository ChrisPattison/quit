#pragma once
#include <functional>
#include <vector>
#include <type_traits>
#include <cassert>
#include <numeric>
#include <iostream>
#include <mpi.h>
#include "parallel_types.hpp"

// TODO: implement async

namespace parallel
{
class Mpi {
private:
    static constexpr int kRoot = 0;
    static constexpr int kVectorHeirarchyBase = 4;
    static constexpr int kScalarHeirarchyBase = 20;
    int world_rank_;
    int world_size_;
    int tag_;
    Heirarchy vector_heirarchy;
    Heirarchy scalar_heirarchy;

    int GetTag();

    void IncrTag(int count);

    int rank(MPI_Comm comm);

    int size(MPI_Comm comm);
    
    bool is_root(MPI_Comm comm);
    // Executes something on root rank
    void ExecRoot(std::function<void()> target, MPI_Comm comm);
    
    template<typename T> auto Reduce(const T& value, std::function<T(std::vector<T>&)> reduce, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;
    
    template<typename T> auto ReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;

    template<typename T> auto VectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>& accumulator, const std::vector<T>& value)> reduce, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;

    template<typename T> auto Gather(const T& value, MPI_Comm comm) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;

    template<typename T> auto AllGather(const T& value, MPI_Comm comm) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;

    template<typename T> auto Send(const T& value, int target_rank, MPI_Comm comm) ->  
    std::enable_if_t<std::is_trivially_copyable<T>::value, void>;

    template<typename T> auto Send(const T& value, int target_rank, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, void>>;

    template<typename T> auto SendAsync(const T& value, int target_rank, MPI_Comm comm) ->  
    std::enable_if_t<std::is_trivially_copyable<T>::value, AsyncOp<T>>;

    template<typename T> auto SendAsync(const T& value, int target_rank, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, AsyncOp<typename T::value_type>>>;

    template<typename T> auto SendAsync(std::vector<T>* value, int target_rank, MPI_Comm comm) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, AsyncOp<T>>;

    template<typename T> auto Receive(int source_rank, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;

    template<typename T> auto Receive(int source_rank, MPI_Comm comm) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, T>>;

    void Barrier(MPI_Comm comm);
    
public:
    Mpi();

    ~Mpi();

    int rank();

    int size();

    bool is_root();

    void ExecRoot(std::function<void()> target) { ExecRoot(target, MPI_COMM_WORLD); }

    template<typename T> auto HeirarchyReduce(const T& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;

    template<typename T> auto PartialGather(const std::vector<Packet<T>>& packets) -> std::vector<T>;

    template<typename T> auto PartialVectorGather(const std::vector<Packet<T>>& packets) -> std::vector<Packet<T>>;

    template<typename T> auto Reduce(const T& value, std::function<T(std::vector<T>&)> reduce) { return Reduce<T>(value, reduce, MPI_COMM_WORLD); }
    
    template<typename T> auto ReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce) { return ReduceToAll<T>(value, reduce, MPI_COMM_WORLD); }

    template<typename T> auto HeirarchyReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;

    template<typename T> auto VectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>& accumulator, const std::vector<T>& value)> reduce) { return VectorReduce<T>(value, reduce, MPI_COMM_WORLD); }

    template<typename T> auto Gather(const T& value) { return Gather<T>(value, MPI_COMM_WORLD); }

    template<typename T> auto AllGather(const T& value) { return AllGather<T>(value, MPI_COMM_WORLD); }

    template<typename T> auto Send(const T& value, int target_rank) { return Send<T>(value, target_rank, MPI_COMM_WORLD); }

    template<typename T> auto SendAsync(const T& value, int target_rank) { return SendAsync<T>(value, target_rank, MPI_COMM_WORLD); }

    template<typename T> auto SendAsync(std::vector<T>* value, int target_rank) { return SendAsync<T>(value, target_rank, MPI_COMM_WORLD); }

    template<typename T> auto Receive(int source_rank) { return Receive<T>(source_rank, MPI_COMM_WORLD); }
    // Make this more discriptive. Version of VectorReduce that scales O(logN)
    template<typename T> auto HeirarchyVectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>& accumulator, const std::vector<T>& value)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;

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

template<typename T> auto Mpi::PartialGather(const std::vector<Packet<T>>& packets) -> std::vector<T> {
    auto data_buffer = PartialVectorGather(packets);
    // combine into one vector / throw out rank information
    int count = std::accumulate(data_buffer.begin(), data_buffer.end(), 0, [](int& acc, auto& packet) {return acc + packet.data.size(); });
    std::vector<T> collated_data;
    collated_data.reserve(count);
    for(auto& data_packet : data_buffer) {
        collated_data.insert(collated_data.end(), data_packet.data.begin(), data_packet.data.end());
    }
    return collated_data;
}

// There may still be a race condition here
template<typename T> auto Mpi::PartialVectorGather(const std::vector<Packet<T>>& packets) -> std::vector<Packet<T>> {
    int tag = GetTag();
    std::vector<MPI_Request> send_requests(packets.size());
    // Send off all data
    for(std::size_t k = 0; k < packets.size(); ++k) {
        MPI_Issend(packets[k].data.data(), packets[k].data.size() * sizeof(T), MPI_BYTE, packets[k].rank, tag, MPI_COMM_WORLD, &send_requests[k]);
    }
    // Find some way to inform targets

    std::vector<MPI_Request> recv_requests;
    MPI_Request barrier_request;
    std::vector<Packet<T>> data_buffer;
    bool barrier_reached = false;
    int all_sent;
    int barrier_done = 0;

    // Busy Wait
    do {
        int message_ready;
        MPI_Message message;
        MPI_Status message_status;
        // Check if incoming request
        MPI_Improbe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &message_ready, &message, &message_status);
        if(message_ready) {
            // Get message size, allocate buffer and receive
            int count;
            data_buffer.emplace_back();
            MPI_Get_count(&message_status, MPI_BYTE, &count);
            assert(count % sizeof(T) == 0);
            data_buffer.back().data.resize(count/sizeof(T));
            data_buffer.back().rank = message_status.MPI_SOURCE;
            recv_requests.emplace_back();
            MPI_Imrecv(data_buffer.back().data.data(), count, MPI_BYTE, &message, &recv_requests.back());
        }
        // Check if all outbound sends completed
        // Warning: Completion of outbound sends does not imply that the reciever has completed the corresponding recv
        MPI_Testall(send_requests.size(), send_requests.data(), &all_sent, MPI_STATUSES_IGNORE);
        if(all_sent && !barrier_reached) {
            MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request); 
            barrier_reached = true;
        }
        // Check if everyone has reached the barrier
        if(barrier_reached) {
            MPI_Test(&barrier_request, &barrier_done, MPI_STATUS_IGNORE);
        }
    } while(!barrier_done);
    
    // Wait for all receives to complete
    MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
    return data_buffer;
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
}