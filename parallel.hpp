#pragma once
#include <functional>
#include <vector>
#include <type_traits>
#include <mpi.h>

// TODO: implement async

class Parallel {
    static constexpr int kRoot = 0;
    int rank_;
    int size_;
    int tag_;
    
    int GetTag();

public:
    Parallel();

    ~Parallel();

    int rank();

    int size();
    
    bool is_root();
    // Executes something on root rank
    void ExecRoot(std::function<void()> target);
    
    template<typename T> auto Reduce(T&& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;
    
    template<typename T> auto ReduceToAll(T&& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;
    
    template<typename T> auto ReduceRootToAll(T&& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, T>;

    template<typename T> auto Reduce(T&& value, std::function<T(std::vector<T>&)> reduce) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, T>>;


    template<typename T> auto Gather(T&& value) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;

    template<typename T> auto AllGather(T&& value) ->
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;

    template<typename T> auto Send(T&& value, int target_rank) ->  
    std::enable_if_t<std::is_trivially_copyable<T>::value, void>;

    template<typename T> auto Send(T&& value, int target_rank) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, void>>;

    template<typename T> auto Receive(int source_rank) -> 
    std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>>;

    template<typename T> auto Receive(int source_rank) -> 
    std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
    std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, T>>;

    void Barrier();
};

template<typename T> auto Parallel::Reduce(T&& value, std::function<T(std::vector<T>&)> reduce) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, T> {
    std::vector<T> data_buffer;
    if(is_root()) {
        data_buffer.resize(size());
    }

    MPI_Gather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, kRoot, MPI_COMM_WORLD);

    if(is_root()) {
        return reduce(data_buffer);
    }else {
        return {};
    }
}

template<typename T> auto Parallel::ReduceRootToAll(T&& value, std::function<T(std::vector<T>&)> reduce) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, T> {
    T reducedvalue = Reduce(value, reduce);

    MPI_BCAST(&reducedvalue, sizeof(T), MPI_BYTE, kRoot, MPI_COMM_WORLD);
    return reducedvalue;
}

template<typename T> auto Parallel::ReduceToAll(T&& value, std::function<T(std::vector<T>&)> reduce) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, T> {
    std::vector<T> data_buffer(sizeof(T) * size());

    MPI_Allgather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, MPI_COMM_WORLD);

    return reduce(data_buffer);
}

template<typename T> auto Parallel::Reduce(T&& value, std::function<T(std::vector<T>&)> reduce) -> 
std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, T>> {
    
    if(is_root()) {
        MPI_Status status;
        MPI_Message message;
        int message_size;
        std::vector<T> data_buffers(size());
        data_buffers.front() = value;
        int tag = GetTag();
        std::vector<MPI_Request> requests(size()-1);
        
        for(std::size_t k = 0; k < size() - 1; ++k) {
            MPI_Mprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &message, &status);
            MPI_Get_count(&status, MPI_BYTE, &message_size);
            data_buffers[status.MPI_SOURCE].resize(message_size/sizeof(typename std::remove_reference_t<decltype(value)>::value_type));
            MPI_Imrecv(data_buffers[status.MPI_SOURCE].data(), message_size, MPI_BYTE, &message, &requests[k]);
        }

        for(std::size_t k = 0; k < requests.size(); ++k) {
            MPI_Wait(&requests[k], MPI_STATUS_IGNORE);
        }

        return reduce(data_buffers);
    }else {
        MPI_Bsend(value.data(), value.size() * sizeof(typename std::remove_reference_t<decltype(value)>::value_type), MPI_BYTE, kRoot, GetTag(), MPI_COMM_WORLD);
        return { };
    }
}

template<typename T> auto Parallel::Gather(T&& value) ->
std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>> {
        std::vector<T> data_buffer;
        if(is_root()) {
            data_buffer.resize(size());
        }
        MPI_Gather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, kRoot, MPI_COMM_WORLD);
        return data_buffer;
}

template<typename T> auto Parallel::AllGather(T&& value) ->
std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>> {
        std::vector<T> data_buffer(size());
        MPI_Allgather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, MPI_COMM_WORLD);
        return data_buffer;
}

template<typename T> auto Parallel::Send(T&& value, int target_rank) ->  
std::enable_if_t<std::is_trivially_copyable<T>::value, void> {
    MPI_Send(&value, sizeof(T), MPI_BYTE, target_rank, 0, MPI_COMM_WORLD);
}

template<typename T> auto Parallel::Send(T&& value, int target_rank) -> 
std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, void>> {
    MPI_Send(value.data(), value.size() * sizeof(typename std::remove_reference_t<decltype(value)>::value_type), MPI_BYTE, target_rank, 0, MPI_COMM_WORLD);
}

template<typename T> auto Parallel::Receive(int source_rank) -> 
std::enable_if_t<std::is_trivially_copyable<T>::value, std::vector<T>> {
    T value;
    MPI_Recv(&value, sizeof(T), MPI_BYTE, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return value;
}

template<typename T> auto Parallel::Receive(int source_rank) -> 
std::enable_if_t<std::is_trivially_copyable<typename T::value_type>::value, 
std::enable_if_t<std::is_same<std::vector<typename T::value_type>, T>::value, T>> {
    MPI_Status status;
    MPI_Message message;
    int message_size;
    T data_buffer;
    
    MPI_Mprobe(source_rank, 0, MPI_COMM_WORLD, &message, &status);
    MPI_Get_count(&status, MPI_BYTE, &message_size);
    data_buffer.resize(message_size);
    MPI_Mrecv(data_buffer.data(), message_size, MPI_BYTE, &message, MPI_STATUS_IGNORE);
    return data_buffer;
}