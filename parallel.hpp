#pragma once
#include <functional>
#include <vector>
#include <type_traits>
#include <cassert>
#include <mpi.h>

// TODO: implement async
// TODO: implement heirarchial reduction

namespace parallel
{
class Heirarchy {
    int levels_;
    int base_;
    std::vector<MPI_Comm> comms_;

    int rank(MPI_Comm comm);
public:
    Heirarchy();
    Heirarchy(const Heirarchy& source) = delete;
    Heirarchy(int base);
    ~Heirarchy();

    const std::vector<MPI_Comm>& comms();
    int global_levels();
    int local_levels();
    int base();
};

template<typename T, typename = std::enable_if_t<std::is_trivially_copyable<T>::value, void>> struct Packet {
    int rank;
    std::vector<T> data;
};

class Mpi {
public:

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
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, void>>;

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

    template<typename T> auto PartialReduce(const std::vector<Packet<T>>& packets) -> std::vector<T>;

    template<typename T> auto Reduce(const T& value, std::function<T(std::vector<T>&)> reduce) { return Reduce<T>(value, reduce, MPI_COMM_WORLD); }
    
    template<typename T> auto ReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce) { return ReduceToAll<T>(value, reduce, MPI_COMM_WORLD); }

    template<typename T> auto HeirarchyReduceToAll(const T& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;

    template<typename T> auto VectorReduce(const std::vector<T>& value, std::function<void (std::vector<T>& accumulator, const std::vector<T>& value)> reduce) { return VectorReduce<T>(value, reduce, MPI_COMM_WORLD); }

    template<typename T> auto Gather(const T& value) { return Gather<T>(value, MPI_COMM_WORLD); }

    template<typename T> auto AllGather(const T& value) { return AllGather<T>(value, MPI_COMM_WORLD); }

    template<typename T> auto Send(const T& value, int target_rank) { return Send<T>(value, target_rank, MPI_COMM_WORLD); }

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

template<typename T> auto Mpi::PartialReduce(const std::vector<Packet<T>>& packets) -> std::vector<T> {
    // make this constant better
    constexpr int kMaxBuffer = 5000;
    std::vector<MPI_Request> requests(packets.size()+2); // Recv and Barrier
    MPI_Status wait_status;
    for(std::size_t k = 0; k < packets.size(); ++k) {
        MPI_Isend(packets[k].data.data(), packets[k].data.size() * sizeof(T), MPI_BYTE, packets[k].rank, 0, MPI_COMM_WORLD, &requests[k]);
    }
    
    std::vector<T> data_buffer(kMaxBuffer);
    bool send_complete = packets.size() == 0;
    bool barrier_reached = false;
    int finished_index;
    do {
        MPI_Irecv(data_buffer.data() + data_buffer.size() - kMaxBuffer, kMaxBuffer * sizeof(T), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &requests[requests.size() - 2]);
        // Handle Send finishes
        do {
            // TODO: make this loop prettier
            if(send_complete && !barrier_reached) {
                MPI_Ibarrier(MPI_COMM_WORLD, &requests[requests.size() - 1]); 
                barrier_reached = true;
            }
            MPI_Waitany(requests.size() - (send_complete ? 0 : 1), requests.data(), &finished_index, &wait_status);
            if(!send_complete) {
                int flag = 0;
                MPI_Testall(requests.size() - 2, requests.data(), &flag, MPI_STATUS_IGNORE);
                if(flag) {
                    send_complete = true;
                }
            }
        } while(finished_index < requests.size() - 2);
        // Recieve Data
        if(finished_index == requests.size() - 2) {
            int count;
            MPI_Get_count(&wait_status, MPI_BYTE, &count);
            data_buffer.insert(data_buffer.end(), count/sizeof(T), T());
        }
    } while(finished_index != requests.size() - 1);
    
    MPI_Cancel(&requests[requests.size() - 2]);
    data_buffer.erase(data_buffer.end() - kMaxBuffer, data_buffer.end());
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
    MPI_Send(value.data(), value.size() * sizeof(typename std::remove_reference_t<decltype(value)>::value_type), MPI_BYTE, target_rank, 0, comm);
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