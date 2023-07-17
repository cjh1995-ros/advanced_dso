#pragma once

#include <opencv2/core/types.hpp>
#include <vector>

namespace adso
{


/// @brief Grid class
/// @tparam T 
template <typename T>
class Grid
{
public:
    using container = std::vector<T>;
    using value_type = typename container::value_type;
    using pointer = typename container::pointer;
    using const_pointer = typename container::const_pointer;
    using referece = typename container::reference;
    using const_referece = typename container::const_reference;
    using iterator = typename container::iterator;
    using const_iterator = typename container::const_iterator;
    using const_reverse_iterator = typename container::const_reverse_iterator;
    using reverse_iterator = typename container::reverse_iterator;
    using size_type = typename container::size_type;
    using difference_type = typename container::difference_type;
    using allocator_type = typename container::allocator_type;

private:
    cv::Size grid_size_{};
    std::vector<T> data_{};

public:
    Grid() = default;
    Grid(int rows, int cols, const T& val = T())
        : grid_size_(rows, cols), data_(rows * cols, val) {}
    Grid(const cv::Size& size, const T& val = T())
        : grid_size_(size), data_(size.area(), val) {}

    /// @brief Reset and resize of grid information
    void reset(const T &val = {}) {data_.assign(data_.size(), val);};
    void resize(const cv::Size &cvsize, const T &val = {})
    {
        grid_size_ = cvsize;
        data_.resize(cvsize.area(), val);
    }

    /// @brief Get the point value in the grid. Define overloading functions for 
    ///       different types of input
    /// @return  
    int rc2ind(int r, int c) const {return r * grid_size_.width + c;}

    T& at(int r, int c) {return data_.at(rc2ind(r, c));}
    const T& at(int r, int c) const {return data_.at(rc2ind(r, c));}

    T& at(const cv::Point &pt) {return at(rc2ind(pt.y, pt.x));}
    const T& at(const cv::Point &pt) const {return at(rc2ind(pt.y, pt.x));}

    T& at(size_t i) {return data_.at(i);}
    const T& at(size_t i) const {return data_.at(i);}


    /// @brief Get the iterator of the grid
    /// @return 
    iterator begin() {return data_.begin();}
    iterator end() {return data_.end();}
    const_iterator begin() const {return data_.begin();}
    const_iterator end() const {return data_.end();}
    const_iterator cbegin() const {return data_.cbegin();}
    const_iterator cend() const {return data_.cend();}

    /// @brief Get the size of grid. 
    /// @return 
    cv::Size cvsize() const noexcept {return grid_size_;}
    int area() const noexcept {return grid_size_.area();}
    bool empty() const noexcept {return data_.empty();}
    size_t size() const noexcept {return data_.size();}

    /// @brief Get the number of rows, columns, width and height of grid
    /// @return 
    int rows() const noexcept {return grid_size_.height;}
    int cols() const noexcept {return grid_size_.width;}
    int width() const noexcept {return grid_size_.width;}
    int height() const noexcept {return grid_size_.height;}


};


} // end of namespace adso