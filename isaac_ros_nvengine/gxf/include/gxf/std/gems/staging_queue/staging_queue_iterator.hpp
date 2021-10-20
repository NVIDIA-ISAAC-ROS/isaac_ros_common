/*
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_STD_GEMS_STAGING_QUEUE_ITERATOR_HPP
#define NVIDIA_GXF_STD_GEMS_STAGING_QUEUE_ITERATOR_HPP

#include <iterator>

namespace gxf {
namespace staging_queue {

// A forward iterator for a staging queue.
//
// This type can be used to iterate through the items of the main stage of a staging queue. (It
// could be used to iterate over any part of the ring buffer, but is only used for the main stage.)
// The iterator correctly "wraps around" the ring buffer in case the iterated sequence spans over
// the end of the ring buffer. This iterator could in theory also be used to loop over the range
// multiple times by adding multiples of size to the end index, however it is not used in this
// fashion by the staging queue.
//
// Let's take the example of a staging queue which stores four items (capacity = 2) as depicted in
// the diagram below. An iterator using index=3 would point to the third element. An iterator using
// index=5 would point to the second element in the sequence. If these two iterators would be used
// as begin and end of a range iteration would "wrap around" and iterate over the items A and B.
//
//    0   1   2   3   4
//    #   #   #   #   #
//    | B | O | O | A |
//        |       |
//        |       # begin
//        # end
//
// Note that the member functions of this type are part of the standard implementation for a
// forward iterator. Please refer to the C++ standard definition for forward iterators for details.
template <typename T>
class StagingQueueIterator {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using reference = const T&;
  using pointer = const T*;

  // Creates an empty (invalid) iterator
  StagingQueueIterator() : data_(nullptr), size_(0), index_(0) {}

  // Creates and iterator based on a sequence of items pointed to by 'data' which stores 'size'
  // number of items. The given 'index' indicates the position of the element in the underlying
  // sequence modulo the length of the sequence.
  StagingQueueIterator(pointer data, size_t size, size_t index)
      : data_(data), size_(size), index_(index) {}

  StagingQueueIterator(const StagingQueueIterator& other) = default;
  StagingQueueIterator& operator=(const StagingQueueIterator& other) = default;
  StagingQueueIterator(StagingQueueIterator&& other) = default;
  StagingQueueIterator& operator=(StagingQueueIterator&& other) = default;

  ~StagingQueueIterator() = default;

  const StagingQueueIterator& operator++() {
    ++index_;
    return *this;
  }
  StagingQueueIterator operator++(int) { return StagingQueueIterator(data_, size_, index_++); }

  reference operator*() const { return data_[index_ % size_]; }
  pointer operator->() const { return data_ + (index_ % size_); }

  bool operator==(const StagingQueueIterator& rhs) const {
    return data_ == rhs.data_ && size_ == rhs.size_ && index_ == rhs.index_;
  }
  bool operator!=(const StagingQueueIterator& rhs) const { return !(*this == rhs); }

 private:
  // A pointer to the sequence of items to which this iterator refers.
  const pointer data_;

  // The length of the sequence pointer to by 'data_'.
  const size_t size_;

  // The index of the item in the sequence to which this iterator points. The index is taken modulo
  // 'size_' when accessing 'data_'.
  size_t index_;
};

}  // namespace staging_queue
}  // namespace gxf

namespace std {

template <typename T>
struct iterator_traits<gxf::staging_queue::StagingQueueIterator<T>> {
  using difference_type = typename gxf::staging_queue::StagingQueueIterator<T>::difference_type;
  using value_type = typename gxf::staging_queue::StagingQueueIterator<T>::value_type;
  using reference = typename gxf::staging_queue::StagingQueueIterator<T>::reference;
  using pointer = typename gxf::staging_queue::StagingQueueIterator<T>::pointer;
  using iterator_category = std::forward_iterator_tag;
};

}  // namespace std

#endif
